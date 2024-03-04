from diffusers import DiffusionPipeline
import torch
from datetime import datetime

pipe = DiffusionPipeline.from_pretrained(
    "playgroundai/playground-v2.5-1024px-aesthetic",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# # Optional: Use DPM++ 2M Karras scheduler for crisper fine details
# from diffusers import EDMDPMSolverMultistepScheduler
# pipe.scheduler = EDMDPMSolverMultistepScheduler()

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# prompt = "In the heart of the ancient forest, the old oak tree whispered secrets of the ages to those who dared to listen. Its leaves rustled with the wisdom of time, telling tales of forgotten worlds and hidden realms."
# create an array to store the prompts
# prompts = ["In the heart of the ancient forest, the old oak tree whispered secrets of the ages to those who dared to listen. Its leaves rustled with the wisdom of time, telling tales of forgotten worlds and hidden realms.", "The anthropologist meticulously analyzed the artifact, deducing its provenance with unparalleled acuity. Her conclusions shed light on the intricacies of the civilization's social hierarchy and religious practices.", "\"I can't believe you're leaving,\" she said, her voice quivering with emotion. \"After all we've been through, how can you just walk away?\" His response was soft but firm, \"It's not about walking away. It's about finding a path that's truly mine.\"", "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy. This transformation occurs in the chloroplasts, utilizing chlorophyll to capture the light.", "Why did the scarecrow win an award? Because he was outstanding in his field! But seriously, he was literally the best at standing still and scaring away the crows. It's a niche talent, but someone's got to do it.", "The sunset painted the sky in hues of orange, pink, and lavender, a canvas of light fading into the serene twilight. Each brushstroke a whisper of the day's end, a promise of the night's peaceful embrace.", "To bake a perfect loaf of bread, start by mixing flour, water, yeast, and salt into a smooth dough. Knead the dough on a lightly floured surface until elastic, then let it rise in a warm place until doubled in size.", "The discovery of penicillin in 1928 by Alexander Fleming marked a turning point in medical history. This accidental discovery revolutionized the treatment of bacterial infections, saving countless lives.", "What is the nature of consciousness? Can machines ever truly possess it, or is it a unique aspect of biological life? These questions challenge our understanding of existence and the essence of being.", "Wander through the cobblestone streets of the old city, where history whispers around every corner. Explore the vibrant markets, the majestic architecture, and the serene parks that offer a tranquil escape from the bustling city life."]

# prompts = ["Wander through the cobblestone streets of the old city, where history whispers around every corner. Explore the vibrant markets, the majestic architecture, and the serene parks that offer a tranquil escape from the bustling city life."]

prompts = [
    "A hyper-realistic image of a dew-covered spider web illuminated by the morning sun.",
    "A fantasy landscape with floating islands, giant mushrooms, and a waterfall flowing into the sky.",
    "A bustling medieval marketplace, with vendors, knights, and townsfolk, set against a castle backdrop.",
    "A depiction of joy at a surprise birthday party, with people in various expressions of happiness and surprise.",
    "An ultra-modern skyscraper design, blending futuristic elements with eco-friendly green spaces.",
    "A serene mountain lake at sunrise, with reflections of the surrounding peaks in the still water.",
    "An abstract painting that visualizes the concept of chaos and order through vibrant colors and geometric shapes.",
    "A detailed portrait of an exotic bird in its natural habitat, showcasing its vivid plumage and textures.",
    "A traditional festival in India, with people celebrating Holi, the festival of colors, in vibrant traditional attire.",
    "A conceptual image of a city in the year 2100, with flying cars, towering green skyscrapers, and holographic signs."
]


# create a for loop to iterate through the prompts array
for prompt in prompts:

    image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=3).images[0]

    # Create the filename string to save it as a .jpg file that starts with the date and time
    # from the datetime module

    filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "ChatGPT-Prompts" + ".jpg"

    # Save the image
    image.save(filename)
