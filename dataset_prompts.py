# All in-code comments are in English as requested.

PROMPT_DATASET = {
    "1_natural_landscape": {
        "base": "A wide landscape photograph of a mountain range under a clear blue sky",
        "variations": [
            "A wide landscape photograph of a mountain range under a stormy night sky",
            "A wide landscape photograph of a mountain range covered in thick white snow",
            "A wide landscape photograph of a mountain range with a wooden cabin in the foreground",
            "An oil painting of a mountain range under a clear blue sky",
            "A wide landscape photograph of a mountain range under a glowing purple sky"
        ]
    },
    "2_close_up_portrait": {
        "base": "A close-up studio portrait of a smiling woman with long brown hair",
        "variations": [
            "A close-up studio portrait of a crying woman with long brown hair",
            "A close-up studio portrait of a smiling woman with long brown hair wearing sunglasses",
            "A close-up studio portrait of a smiling old woman with long brown hair",
            "A close-up studio portrait of a smiling woman with long brown hair in neon cyberpunk lighting",
            "A close-up studio portrait of a smiling woman with long bright green hair"
        ]
    },
    "3_still_life": {
        "base": "A high-quality photograph of a red apple resting on a rustic wooden table",
        "variations": [
            "A high-quality photograph of a glass bottle resting on a rustic wooden table",
            "A high-quality photograph of a red apple resting on a shiny marble table",
            "A high-quality photograph of a black apple resting on a rustic wooden table",
            "A high-quality photograph of a red apple and a sharp knife resting on a rustic wooden table",
            "A high-quality photograph of a sliced apple resting on a rustic wooden table"
        ]
    },
    "4_animal_portrait": {
        "base": "A cinematic shot of a golden retriever dog sitting on a green lawn",
        "variations": [
            "A cinematic shot of a black cat sitting on a green lawn",
            "A cinematic shot of a golden retriever dog sleeping on a green lawn",
            "A cinematic shot of a golden retriever dog sitting on a sandy beach",
            "A cinematic shot of a golden retriever dog wearing a red collar sitting on a green lawn",
            "A watercolor painting of a golden retriever dog sitting on a green lawn"
        ]
    },
    "5_urban_architecture": {
        "base": "A street-level view of a modern skyscraper building in a busy downtown area",
        "variations": [
            "A street-level view of a modern skyscraper building at sunset",
            "A street-level view of a gothic cathedral building in a busy downtown area",
            "A street-level view of a modern skyscraper building on a heavy rainy day",
            "A street-level view of an abandoned ruined skyscraper building in a busy downtown area",
            "A street-level view of a bright yellow skyscraper building in a busy downtown area"
        ]
    },
    "6_food_photography": {
        "base": "A top-down photograph of a white ceramic plate with a fresh chocolate cake",
        "variations": [
            "A top-down photograph of a white ceramic plate with a cheese pizza",
            "A top-down photograph of a white ceramic plate with a fresh chocolate cake topped with strawberries",
            "A top-down photograph of a golden metal plate with a fresh chocolate cake",
            "A top-down photograph of a white ceramic plate with a half-eaten chocolate cake",
            "A top-down photograph on a dark stone surface of a white ceramic plate with a fresh chocolate cake"
        ]
    },
    "7_vehicles": {
        "base": "A profile shot of a sleek sports car parked on an empty highway",
        "variations": [
            "A profile shot of a rusty old truck parked on an empty highway",
            "A profile shot of a bright pink sports car parked on an empty highway",
            "A profile shot of a sleek sports car parked on an empty highway under foggy morning mist",
            "A profile shot of a burning sports car parked on an empty highway",
            "A profile shot of a sleek sports car parked in a futuristic spaceship hangar"
        ]
    },
    "8_interior_design": {
        "base": "A wide-angle photo of a minimalist living room with a white sofa",
        "variations": [
            "A wide-angle photo of a chaotic vintage living room with a white sofa",
            "A wide-angle photo of a minimalist living room with a dark leather sofa",
            "A wide-angle photo of a minimalist living room with a white sofa and a roaring fireplace",
            "A wide-angle photo of a minimalist living room with a white sofa illuminated by moonlight through the window",
            "A wide-angle photo of a minimalist living room with a transparent glass sofa"
        ]
    },
    "9_fantasy_sci_fi": {
        "base": "A detailed illustration of a glowing crystal sword floating in the air",
        "variations": [
            "A detailed illustration of a glowing crystal shield floating in the air",
            "A detailed illustration of a dark shadowy sword floating in the air",
            "A 3D digital render of a glowing crystal sword floating in the air",
            "A detailed illustration of a glowing crystal sword embedded inside a giant rock",
            "A detailed illustration of a glowing crystal sword covered in burning flames floating in the air"
        ]
    },
    "10_abstract_graphic": {
        "base": "A minimalist vector art pattern of sharp geometric triangles",
        "variations": [
            "A minimalist vector art pattern of soft geometric circles",
            "A black and white vector art pattern of sharp geometric triangles",
            "A 3D neon wireframe pattern of sharp geometric triangles",
            "A highly complex and chaotic vector art pattern of sharp geometric triangles",
            "A minimalist vector art pattern of sharp metallic triangles"
        ]
    }
}

# Example of how to access the data in your generation loop:
if __name__ == "__main__":
    for category, content in PROMPT_DATASET.items():
        print(f"Category: {category}")
        print(f"  Base Prompt: {content['base']}")
        for i, var in enumerate(content['variations']):
            print(f"  Variation {i+1}: {var}")
        print("-" * 40)

