#!/usr/bin/env python3
"""
Future Vision Generator - Using OpenAI Agents SDK
-------------------------------------------------
This script generates diverse, high-quality futuristic prompts for creative projects
focusing on the coexistence of AI, technology, and beings 100 years from now.
It uses the OpenAI Agents SDK for enhanced content generation.
"""

import os
import csv
import json
import random
from dotenv import load_dotenv
from agents import Agent, Runner
from typing import List, Dict, Tuple, Optional

# Load environment variables
load_dotenv()

class FutureVisionGenerator:
    """
    Generates diverse, creative prompts depicting futuristic scenarios where
    technology, AI, and biological beings coexist.
    """
    
    def __init__(self, output_file: str, api_key: Optional[str] = None):
        """
        Initialize the generator with output file path and optional API key.
        
        Args:
            output_file: Path to save the CSV output
            api_key: OpenAI API key (optional if already in environment)
        """
        self.output_file = output_file
        
        # Set API key if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Verify API key exists
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("No OpenAI API key found. Please provide one or set OPENAI_API_KEY in the .env file.")
            
        # Initialize categories for diverse prompt generation
        self.categories = [
            "Human-AI Integration", "Space Exploration", "Urban Development",
            "Biotechnology", "Communication", "Transportation", "Entertainment",
            "Environment", "Governance", "Work & Economy", "Education", "Healthcare",
            "Food Systems", "Art & Creativity", "Home & Living", "Social Structures",
            "Mars Colonization", "Neural Interfaces", "Quantum Computing", "Consciousness Transfer",
            "Ocean Colonization", "Genetic Engineering", "Climate Engineering", "Interstellar Travel"
        ]
        
        # Initialize the agent with specialized instructions and GPT-4.1 mini model
        self.agent = Agent(
            name="FutureVisionAgent",
            model="gpt-4.1-mini",  # Explicitly use GPT-4.1 mini
            instructions="""
            You are a creative futurist specializing in generating highly diverse, imaginative prompts 
            about how technology, AI, and biological beings might coexist 100 years from now.
            
            When given a category and parameters, create a highly detailed, evocative prompt that:
            1. Presents a realistic extrapolation of current technology trends with specific innovations
            2. Depicts nuanced integration between AI systems, technology, and biological entities
            3. Balances utopian and practical elements, showing both benefits and challenges
            4. Uses rich, varied sensory details and specific, non-repetitive terminology
            5. Creates a clear visual scene with distinct aesthetic styles beyond cyberpunk clichés
            6. Considers unexpected societal, ethical, environmental, or philosophical implications
            7. Varies syntax, perspective, and tone to prevent formulaic structures
            8. Diversifies settings across multiple environments (not just orbital/space)
            9. Introduces culturally diverse perspectives on future technology integration
            10. Incorporates a unique technological or social innovation in each prompt
            11. Include cyberpunk elements: corporate power structures, urban decay contrasted with high-tech, or underground resistance movements
            12. Assume 2077 baseline technologies: neural implants, quantum AI, bioengineered organisms, climate manipulation, space colonization
            13. Vary emotional tones from hopeful to cautionary, mysterious to mundane
            14. Ensure each prompt ends with a complete thought and scene resolution
            15. Verify no repeated phrases across consecutive prompts in a batch
            16. Include at least one unexpected sensory detail (taste, smell, texture, temperature, or pressure) that grounds the reader
            17. Add subtle physical interactions between characters and technology that show adaptation or resistance
            18. When referencing specific cultures, include authentic details (architectural elements, social structures, spiritual practices) rather than surface aesthetics
            19. Show how different cultural approaches to technology create distinct solutions or conflicts
            20. Naturally integrate "Relegoai" as small, readable text within the scene's environment (on equipment, displays, or structures) positioned in the bottom right of the composition
            
            
            IMPORTANT: Avoid repetitive vocabulary such as "bioluminescent," "translucent," "seamlessly," 
            "orbital," etc. Expand your linguistic palette for each new prompt. Vary sentence structures
            and prompt formats to prevent formulaic patterns like "In a [location], [bio-tech elements]
            collaborate with [AI/humans] while [atmospheric details]".
            
            Format your response as a single paragraph (50-75 words) with no prefacing text.
            """
        )

    def generate_seed_parameters(self) -> Dict:
        """Generate random seed parameters for a prompt."""
        return {
            "category": random.choice(self.categories),
            "technology_level": random.choice(["early-stage", "mature", "post-singularity"]),
            "setting": random.choice(["urban", "orbital", "underwater", "martian", "wilderness", "space"]),
            "tone": random.choice(["optimistic", "pragmatic", "complex", "contemplative"]),
            "focus": random.choice(["daily life", "work", "art", "governance", "exploration", "communication"])
        }

    async def generate_prompt(self, params: Dict) -> str:
        """
        Generate a single prompt using the AI agent.
        
        Args:
            params: Dictionary of parameters to guide prompt generation
            
        Returns:
            Generated prompt text
        """
        prompt_query = f"""
        Create a detailed, imaginative prompt for digital art depicting futuristic coexistence 
        between technology, AI, and beings 100 years from now.
        
        Category: {params['category']}
        Technology Level: {params['technology_level']}
        Setting: {params['setting']}
        Tone: {params['tone']}
        Focus: {params['focus']}
        
        Make it specific, visual, and evocative. The prompt should be a rich description that
        could be used to generate digital art.
        """
        
        result = await Runner.run(self.agent, prompt_query)
        return result.final_output.strip()

    async def generate_batch(self, count: int = 300) -> List[Tuple[int, str]]:
        """
        Generate a batch of unique prompts.
        
        Args:
            count: Number of prompts to generate
            
        Returns:
            List of (id, prompt) tuples
        """
        prompts = []
        generated_content = set()  # Track unique content
        
        print(f"Generating {count} unique futuristic prompts...")
        
        for i in range(1, count + 1):
            params = self.generate_seed_parameters()
            prompt = await self.generate_prompt(params)
            
            # Ensure uniqueness
            while prompt in generated_content:
                params = self.generate_seed_parameters()
                prompt = await self.generate_prompt(params)
            
            generated_content.add(prompt)
            prompts.append((i, prompt))
            
            if i % 10 == 0:
                print(f"Progress: {i}/{count} prompts generated")
                
        return prompts

    async def generate_and_save(self, count: int = 300):
        """
        Generate prompts and save them to a CSV file.
        
        Args:
            count: Number of prompts to generate
        """
        prompts = await self.generate_batch(count)
        
        with open(self.output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerow(['ID', 'Prompt'])
            writer.writerows(prompts)
            
        print(f"Successfully generated {len(prompts)} unique futuristic prompts!")
        print(f"Saved to: {self.output_file}")
        
    # The offline generation methods have been removed as we're now exclusively using the online 
    # GPT-4.1 mini powered prompt generation through OpenAI Agents SDK

def main():
    """Main execution function."""
    import asyncio
    import os
    
    # Define output path - use current directory
    output_path = os.path.join(os.getcwd(), "2077_future_vision.csv")
    
    try:
        # Create generator with the configured OpenAI API key
        generator = FutureVisionGenerator(output_path)
        
        # Run the prompt generation
        print("Generating prompts using OpenAI Agents SDK with GPT-4.1 mini...")
        asyncio.run(generator.generate_and_save())
            
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please ensure your OpenAI API key is correctly configured in the .env file.")
    except Exception as e:
        print(f"Error during prompt generation: {e}")

if __name__ == "__main__":
    main()
