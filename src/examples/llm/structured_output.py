"""
Structured Output Example

Demonstrates:
- Pydantic models for LLM output
- Instructor library for structured extraction
- Function calling
- Validation and error handling
- Multiple output schemas

Usage:
    export OPENAI_API_KEY="sk-..."

    python -m src.examples.llm.structured_output
"""

import asyncio
import os
from enum import Enum
from typing import Any

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator

# Output Schemas


class Sentiment(str, Enum):
    """Sentiment classification."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentAnalysis(BaseModel):
    """Structured sentiment analysis result."""

    text: str = Field(description="The analyzed text")
    sentiment: Sentiment = Field(description="Overall sentiment")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(description="Brief explanation of the classification")


class Person(BaseModel):
    """Extracted person information."""

    name: str = Field(description="Full name")
    age: int | None = Field(None, description="Age if mentioned")
    occupation: str | None = Field(None, description="Occupation if mentioned")
    location: str | None = Field(None, description="Location if mentioned")


class EntityExtraction(BaseModel):
    """Structured entity extraction result."""

    people: list[Person] = Field(default_factory=list, description="Extracted people")
    organizations: list[str] = Field(default_factory=list, description="Extracted organizations")
    locations: list[str] = Field(default_factory=list, description="Extracted locations")
    dates: list[str] = Field(default_factory=list, description="Extracted dates")


class TaskPriority(str, Enum):
    """Task priority levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Task(BaseModel):
    """A single task."""

    title: str = Field(description="Task title")
    description: str = Field(description="Detailed description")
    priority: TaskPriority = Field(description="Task priority")
    estimated_hours: float = Field(gt=0, description="Estimated hours to complete")
    dependencies: list[str] = Field(default_factory=list, description="Task dependencies")

    @field_validator("estimated_hours")
    @classmethod
    def validate_hours(cls, v: float) -> float:
        """Validate estimated hours are reasonable."""
        if v > 40:
            raise ValueError("Estimated hours cannot exceed 40")
        return v


class ProjectPlan(BaseModel):
    """Structured project plan."""

    project_name: str = Field(description="Project name")
    description: str = Field(description="Project description")
    tasks: list[Task] = Field(description="List of tasks")
    total_estimated_hours: float = Field(description="Total estimated hours")

    @field_validator("total_estimated_hours")
    @classmethod
    def validate_total_hours(cls, v: float, info: Any) -> float:
        """Validate total hours match sum of task hours."""
        if "tasks" in info.data:
            calculated_total = sum(task.estimated_hours for task in info.data["tasks"])
            if abs(v - calculated_total) > 0.1:
                raise ValueError(f"Total hours {v} doesn't match sum of tasks {calculated_total}")
        return v


class StructuredOutputClient:
    """
    Client for structured LLM output using Instructor.

    Demonstrates:
    - Type-safe LLM responses
    - Automatic validation
    - Retry on validation errors
    - Multiple output schemas
    """

    def __init__(self) -> None:
        """Initialize structured output client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")

        # Patch OpenAI client with Instructor
        self.client = instructor.from_openai(AsyncOpenAI(api_key=api_key))
        self.model = "gpt-4o-mini"

    async def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Structured sentiment analysis
        """
        return await self.client.chat.completions.create(
            model=self.model,
            response_model=SentimentAnalysis,
            messages=[
                {
                    "role": "system",
                    "content": "You are a sentiment analysis expert.",
                },
                {
                    "role": "user",
                    "content": f"Analyze the sentiment of this text: {text}",
                },
            ],
        )

    async def extract_entities(self, text: str) -> EntityExtraction:
        """
        Extract named entities from text.

        Args:
            text: Text to extract from

        Returns:
            Structured entity extraction
        """
        return await self.client.chat.completions.create(
            model=self.model,
            response_model=EntityExtraction,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at extracting named entities.",
                },
                {
                    "role": "user",
                    "content": f"Extract all entities from this text: {text}",
                },
            ],
        )

    async def create_project_plan(self, description: str) -> ProjectPlan:
        """
        Generate structured project plan.

        Args:
            description: Project description

        Returns:
            Structured project plan with tasks
        """
        return await self.client.chat.completions.create(
            model=self.model,
            response_model=ProjectPlan,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a project planning expert. Create detailed, realistic plans."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Create a project plan for: {description}",
                },
            ],
            max_retries=3,  # Retry on validation errors
        )


async def main() -> None:
    """Run example demonstrations."""
    print("=" * 60)
    print("Structured Output Example")
    print("=" * 60)

    client = StructuredOutputClient()

    # Example 1: Sentiment Analysis
    print("\n1. Sentiment Analysis")
    print("-" * 60)

    texts = [
        "This product is absolutely amazing! Best purchase ever!",
        "Terrible experience. Would not recommend to anyone.",
        "It's okay, nothing special but does the job.",
    ]

    for text in texts:
        result = await client.analyze_sentiment(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result.sentiment.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reasoning: {result.reasoning}")

    # Example 2: Entity Extraction
    print("\n\n2. Entity Extraction")
    print("-" * 60)

    text = """
    John Smith, a 35-year-old software engineer at Google,
    recently moved to San Francisco. He met with Sarah Johnson,
    CEO of TechCorp, on January 15th, 2024 to discuss a partnership
    between Google and TechCorp in New York.
    """

    entities = await client.extract_entities(text)

    print(f"\nText: {text.strip()}")
    print("\nPeople:")
    for person in entities.people:
        print(f"  - {person.name}", end="")
        if person.age:
            print(f" (age {person.age})", end="")
        if person.occupation:
            print(f", {person.occupation}", end="")
        if person.location:
            print(f", {person.location}", end="")
        print()

    print(f"\nOrganizations: {', '.join(entities.organizations)}")
    print(f"Locations: {', '.join(entities.locations)}")
    print(f"Dates: {', '.join(entities.dates)}")

    # Example 3: Project Planning
    print("\n\n3. Project Planning")
    print("-" * 60)

    project_desc = "Build a RAG-based chatbot with FastAPI backend and React frontend"

    plan = await client.create_project_plan(project_desc)

    print(f"\nProject: {plan.project_name}")
    print(f"Description: {plan.description}")
    print(f"Total Estimated Hours: {plan.total_estimated_hours}")
    print("\nTasks:")

    for i, task in enumerate(plan.tasks, 1):
        print(f"\n{i}. {task.title} [{task.priority.value.upper()}]")
        print(f"   Description: {task.description}")
        print(f"   Estimated: {task.estimated_hours}h")
        if task.dependencies:
            print(f"   Dependencies: {', '.join(task.dependencies)}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- All outputs are type-safe Pydantic models")
    print("- Automatic validation ensures data quality")
    print("- Instructor handles retries on validation errors")
    print("- Complex nested structures are supported")


if __name__ == "__main__":
    asyncio.run(main())
