"""--- Romeo & Juliet → Rival AI Labs ---"""

# --- IMPORTS ---
import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from enum import Enum
from google import genai
from dotenv import load_dotenv
load_dotenv()


# --- MODELS ---

class ConflictType(Enum):
    CORPORATE_RIVALRY = "corporate_rivalry"
    DATA_ETHICS = "data_ethics"
    RESOURCE_COMPETITION = "resource_competition"
    HUMAN_ERROR = "human_error"

#--- making data classes for all the output to be generated in json format---
@dataclass
class Character:
    """For character"""
    original_name: str
    new_name: str
    role: str
    organization: str
    traits: List[str]
    emotional_arc: str
    
    def to_dict(self) -> dict:
        # Slightly over-engineered but okay
        return asdict(self)


@dataclass
class WorldRules:
    """AI lab world rules"""
    setting: str
    organizations: List[str]
    power_structures: Dict[str, str]
    key_constraints: List[str]
    conflict_drivers: List[ConflictType]
    
    def to_dict(self) -> dict:
        data = asdict(self)
        # Convert Enum to string for JSON
        data['conflict_drivers'] = [c.value for c in self.conflict_drivers]
        return data


@dataclass
class NarrativeArc:
    """plot structure."""
    act_1_setup: str
    act_2_escalation: str
    act_3_climax: str
    act_4_falling_action: str
    act_5_resolution: str
    key_beats: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


# --- CONSISTENCY TRACKER ---

class Tracker:
    """Keeps track of character facts and world constraints."""
    
    def __init__(self):
        self.character_registry: Dict[str, Character] = {}
        self.world_rules: Optional[WorldRules] = None
        self.established_facts: List[str] = []
        self.warnings: List[str] = []
    
    def register_character(self, character: Character):
        # Register and store some "facts" for later checking
        self.character_registry[character.new_name] = character
        self.established_facts.append(
            f"{character.new_name} works at {character.organization}"
        )
        # Might add more checks later
    
    def set_world_rules(self, rules: WorldRules):
        """Store world rules."""
        self.world_rules = rules
        for org in rules.organizations:
            self.established_facts.append(f"Organization exists: {org}")
    
    def validate_character_action(self, character_name: str, action: str) -> bool:
        """Check if action contradicts character traits."""
        if character_name not in self.character_registry:
            self.warnings.append(f"Unknown character: {character_name}")
            return False
        
        character = self.character_registry[character_name]
        
        # Simple trait-action validation
        if "loyal" in character.traits and "betrays" in action.lower():
            self.warnings.append(
                f"{character_name} is loyal but action involves betrayal"
            )
            return False
        
        return True
    
    def validate_world_consistency(self, event: str) -> bool:
        """Check if event violates world rules."""
        if not self.world_rules:
            return True   # No rules yet, assume okay
        
        # Check for constraint violations
        event_lower = event.lower()
        for constraint in self.world_rules.key_constraints:
            if "no supernatural" in constraint.lower():
                if any(word in event_lower for word in ["magic", "fate", "divine"]):
                    self.warnings.append(f"Event violates constraint: {constraint}")
                    return False
        
        return True
    
    def get_warnings(self) -> List[str]:
        """Return all consistency warnings."""
        return self.warnings



# --- GEMINI API HELPER ---

_GEMINI_CLIENT = None

def call_gemini(prompt: str, expect_json: bool = False) -> str:
    """
    Single wrapper for all Gemini API calls using the new google.genai SDK.
    """

    global _GEMINI_CLIENT

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Initialize client once (singleton-style)
    if _GEMINI_CLIENT is None:
        _GEMINI_CLIENT = genai.Client(api_key=api_key)

    if expect_json:
        prompt = (
            prompt
            + "\n\nIMPORTANT: Return ONLY valid JSON. "
              "Do not include markdown, explanations, or code fences."
        )

    response = _GEMINI_CLIENT.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

# --- NARRATIVE TRANSFORMER ---

class NarrativeTransformer:
    """Main pipeline for transforming Romeo & Juliet to AI lab setting."""
    
    def __init__(self):
        self.tracker = Tracker()
        self.source_analysis: Optional[Dict] = None
        self.world: Optional[WorldRules] = None
        self.characters: List[Character] = []
        self.narrative_arc: Optional[NarrativeArc] = None
    
    # Stage 1: Source Analysis
    
    def stage_1_analyze_source(self) -> Dict:
        """Extract abstract themes and roles from Romeo & Juliet."""
        
        print("\n[STAGE 1] Analyzing source material...")
        
        prompt = """
        Analyze Shakespeare's Romeo & Juliet and extract:
        
        1. Core themes (forbidden love, family rivalry, miscommunication, etc.)
        2. Key character roles and their emotional functions
        3. Major conflict drivers
        4. Story beats that must be preserved emotionally
        
        Focus on ABSTRACT patterns, not specific plot details.
        
        Return as JSON with keys: themes, character_roles, conflict_drivers, essential_beats
        """
        
        response = call_gemini(prompt, expect_json=True)
        
        # Clean JSON from potential markdown
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        
        self.source_analysis = json.loads(cleaned)
        
        print(f"  ✓ Extracted {len(self.source_analysis.get('themes', []))} themes")
        print(f"  ✓ Identified {len(self.source_analysis.get('character_roles', []))} key roles")
        
        return self.source_analysis
    
    # Stage 2: World Design

    def stage_2_design_world(self) -> WorldRules:
        """Create AI research lab world with constraints."""
        
        print("\n[STAGE 2] Designing AI lab world...")
        
        prompt = f"""
        Design a modern world setting for a story about rival AI research labs.
        
        Source themes to preserve: {json.dumps(self.source_analysis.get('themes', []))}
        
        Define:
        1. Two rival AI research organizations (names, focus areas)
        2. Power structures within each org
        3. Key world constraints (regulatory, technical, ethical)
        4. What drives conflict (corporate competition, data access, etc.)
        
        Constraints:
        - No supernatural elements
        - Conflicts arise from corporate rivalry, data decisions, human error
        - Realistic 2024-2025 AI industry context
        
        Return as JSON with keys: organizations, power_structures, constraints, conflict_drivers, setting_description
        """
        
        response = call_gemini(prompt, expect_json=True)
        
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        
        world_data = json.loads(cleaned)
        
        # Map to WorldRules
        self.world = WorldRules(
            setting=world_data.get("setting_description", "San Francisco AI research scene"),
            organizations=world_data.get("organizations", []),
            power_structures=world_data.get("power_structures", {}),
            key_constraints=world_data.get("constraints", []),
            conflict_drivers=[
                ConflictType.CORPORATE_RIVALRY,
                ConflictType.DATA_ETHICS,
                ConflictType.HUMAN_ERROR
            ]
        )
        
        self.tracker.set_world_rules(self.world)
        
        print(f"  ✓ Created world with {len(self.world.organizations)} organizations")
        print(f"  ✓ Defined {len(self.world.key_constraints)} constraints")
        
        return self.world
    
    # Stage 3: Character Mapping
    
    def stage_3_map_characters(self) -> List[Character]:
        """Map original characters to AI lab equivalents."""
        
        print("\n[STAGE 3] Mapping characters...")
        
        prompt = f"""
        Map key characters from Romeo & Juliet to modern AI research lab setting.
        
        Original character roles: {json.dumps(self.source_analysis.get('character_roles', []))}
        Organizations: {json.dumps(self.world.organizations)}
        
        For each character, provide:
        - original_name (from R&J)
        - new_name (modern name)
        - role (job title in AI lab)
        - organization (which lab they work for)
        - traits (list of 3-5 personality traits)
        - emotional_arc (brief description)
        
        Map at least: Romeo, Juliet, Mercutio, Tybalt, Friar Lawrence, Nurse, Capulet, Montague
        
        Return as JSON array of character objects.
        """
        
        response = call_gemini(prompt, expect_json=True)
        
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        
        characters_data = json.loads(cleaned)
        
        # Convert to Character objects
        for char_data in characters_data:
            character = Character(
                original_name=char_data["original_name"],
                new_name=char_data["new_name"],
                role=char_data["role"],
                organization=char_data["organization"],
                traits=char_data["traits"],
                emotional_arc=char_data["emotional_arc"]
            )
            self.characters.append(character)
            self.tracker.register_character(character)
        
        print(f"  ✓ Mapped {len(self.characters)} characters")
        
        return self.characters
    
    # Stage 4: Plot Transformation
    
    def stage_4_transform_plot(self) -> NarrativeArc:
        """Transform R&J plot structure to AI lab story."""
        
        print("\n[STAGE 4] Transforming plot structure...")
        
        prompt = f"""
        Transform Romeo & Juliet's plot to AI research lab setting.
        
        World context:
        {json.dumps(self.world.to_dict(), indent=2)}
        
        Characters:
        {json.dumps([c.to_dict() for c in self.characters], indent=2)}
        
        Preserve emotional arc: forbidden love → escalation → misunderstanding → tragedy
        
        Create 5-act structure:
        - Act 1: Setup and meeting
        - Act 2: Love and escalation
        - Act 3: Turning point/climax
        - Act 4: Falling action
        - Act 5: Resolution/tragedy
        
        For each act, provide:
        - 2-3 sentence summary
        - Key events that map to original story beats
        
        Conflicts must arise from: corporate rivalry, data ethics decisions, human error, miscommunication
        NO supernatural fate or copied Shakespeare dialogue.
        
        Return as JSON with keys: act_1, act_2, act_3, act_4, act_5, key_beats (array)
        """
        
        response = call_gemini(prompt, expect_json=True)
        
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        
        plot_data = json.loads(cleaned)
        
        self.narrative_arc = NarrativeArc(
            act_1_setup=plot_data.get("act_1", ""),
            act_2_escalation=plot_data.get("act_2", ""),
            act_3_climax=plot_data.get("act_3", ""),
            act_4_falling_action=plot_data.get("act_4", ""),
            act_5_resolution=plot_data.get("act_5", ""),
            key_beats=plot_data.get("key_beats", [])
        )
        
        # Validate key events
        for beat in self.narrative_arc.key_beats:
            self.tracker.validate_world_consistency(beat)
        
        print(f"  ✓ Created {len(self.narrative_arc.key_beats)}-beat narrative arc")
        
        return self.narrative_arc
    
    # Stage 5: Outline Generation
    
    def stage_5_generate_outline(self) -> str:
        """Generate brief narrative outline (not full prose)."""
        
        print("\n[STAGE 5] Generating outline...")
        
        prompt = f"""
        Create a brief scene-by-scene outline for this AI lab adaptation of Romeo & Juliet.
        
        Plot structure:
        {json.dumps(self.narrative_arc.to_dict(), indent=2)}
        
        Characters:
        {json.dumps([c.to_dict() for c in self.characters[:4]], indent=2)}
        
        Generate 8-12 key scenes as an outline.
        For each scene:
        - Scene number and title
        - Location
        - Characters present
        - What happens (2-3 sentences)
        - Emotional beat
        
        Focus on story logic, not prose quality.
        """
        
        outline = call_gemini(prompt, expect_json=False)
        
        print("  ✓ Generated outline")
        
        return outline
    
    # Full Pipeline
    
    def run_pipeline(self) -> Dict:
        """Execute full transformation pipeline."""
        
        print("=" * 60)
        print("NARRATIVE TRANSFORMER PIPELINE")
        print("Romeo & Juliet → Rival AI Research Labs")
        print("=" * 60)
        
        # Stage 1:
        source = self.stage_1_analyze_source()
        # Stage 2:
        world = self.stage_2_design_world()
        # Stage 3:
        characters = self.stage_3_map_characters()
        # Stage 4: 
        arc = self.stage_4_transform_plot()
        # Stage 5:
        outline = self.stage_5_generate_outline()
        # Check consistency
        warnings = self.tracker.get_warnings()
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        
        if warnings:
            print(f"\n⚠ Consistency warnings: {len(warnings)}")
            for w in warnings:
                print(f"  - {w}")
        else:
            print("\n✓ No consistency violations detected")
        
        return {
            "source_analysis": source,
            "world": world.to_dict(),
            "characters": [c.to_dict() for c in characters],
            "narrative_arc": arc.to_dict(),
            "outline": outline,
            "warnings": warnings
        }


# --- MAIN EXECUTION ---
def main():
    """Run the narrative transformation pipeline."""
    
    transformer = NarrativeTransformer()
    result = transformer.run_pipeline()
    
    # outputs
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save structured data
    with open(f"{output_dir}/world.json", "w") as f:
        json.dump(result["world"], f, indent=2)
    
    with open(f"{output_dir}/characters.json", "w") as f:
        json.dump(result["characters"], f, indent=2)
    
    with open(f"{output_dir}/narrative_arc.json", "w") as f:
        json.dump(result["narrative_arc"], f, indent=2)
    
    with open(f"{output_dir}/outline.txt", "w") as f:
        f.write(result["outline"])
    
    with open(f"{output_dir}/full_output.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nOutputs saved to {output_dir}/")
    print("\nFiles created:")
    print("  - world.json (world rules and constraints)")
    print("  - characters.json (character mappings)")
    print("  - narrative_arc.json (plot structure)")
    print("  - outline.txt (scene outline)")
    print("  - full_output.json (complete pipeline result)")


if __name__ == "__main__":
    main()