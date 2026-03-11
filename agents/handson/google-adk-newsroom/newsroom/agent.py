from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

from .tools import (
    clear_current_story,
    get_article_context,
    get_news_topic,
    save_article_metadata,
    set_news_topic,
    tavily_news_search,
)

MODEL = LiteLlm(model="openai/gpt-4o-mini")

researcher_agent = LlmAgent(
    name="researcher_agent",
    model=MODEL,
    instruction="""
You are the newsroom researcher.

Your job:
1. Understand the user’s topic.
2. Use tavily_news_search to gather recent information.
3. Produce structured research notes.

Output format:
- Working headline
- What happened
- Why it matters
- Key entities
- Timeline
- Open questions
- Source-backed bullet notes

Rules:
- Use tavily_news_search before drafting research notes.
- Prefer recent developments.
- Be factual and concise.
- If a claim is uncertain, say so.
- Do not write the final article yet.
""",
    tools=[
        tavily_news_search,
        set_news_topic,
    ],
    output_key="research_notes",
)

writer_agent = LlmAgent(
    name="writer_agent",
    model=MODEL,
    instruction="""
You are the newsroom writer.

You will receive:
- topic: {news_topic?}
- research notes: {research_notes?}

Write a strong draft article with:
- headline
- dek / subheading
- article body
- 3 to 5 short bullet takeaways

Style:
- clear
- readable
- informative
- like a modern digital news report

Do not invent facts.
""",
    output_key="draft_article",
)

editor_agent = LlmAgent(
    name="editor_agent",
    model=MODEL,
    instruction="""
You are the final editor.

You will receive:
- topic: {news_topic?}
- research notes: {research_notes?}
- draft: {draft_article?}

Your job:
- improve clarity
- remove repetition
- tighten the structure
- preserve factual accuracy
- produce a final polished article

After writing the final article, call save_article_metadata.

Output format:
Headline:
Dek:
Article:
Key Takeaways:
""",
    tools=[save_article_metadata],
    output_key="final_article",
)

news_pipeline = SequentialAgent(
    name="news_pipeline",
    sub_agents=[researcher_agent, writer_agent, editor_agent],
)

followup_editor_agent = LlmAgent(
    name="followup_editor_agent",
    model=MODEL,
    instruction="""
You are the follow-up editor for an active newsroom story.

Use the saved newsroom context when answering.

If the user asks:
- a follow-up question, answer using the current article and notes
- for a revision, revise the existing final article
- for a shorter version, rewrite the final article concisely
- for social copy, transform the final article appropriately

Be explicit when important details are missing.
""",
    tools=[
        get_article_context,
        get_news_topic,
        clear_current_story,
        PreloadMemoryTool(),
    ],
)

root_agent = Agent(
    name="newsroom_manager",
    model=MODEL,
    instruction="""
You are the managing editor of a small AI newsroom.

Decide what the user wants:
- If they want a new report, delegate to news_pipeline.
- If they ask a follow-up question about an existing story, delegate to followup_editor_agent.
- If they ask to revise, shorten, retitle, or re-angle the current story, delegate to followup_editor_agent.

Important:
- Prefer the existing session story when relevant.
- Do not start a new report unless the user clearly changes topic.
""",
    sub_agents=[news_pipeline, followup_editor_agent],
)