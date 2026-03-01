import json
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from state import ResearchState
from tools import ddg, wiki
from langgraph.graph import StateGraph, START,END

llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)

SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages(
    [ ("system", "You are the supervisor of a research team.\n" "Decide the next step: web, wiki, synthesize, done.\n" "Rules: start with wiki/web if notes are empty; synthesize after 3-6 solid notes.\n" "Return ONLY JSON: {{\"next\": \"...\"}}."),
    ("human", "Question: {question}\n\nCurrent notes (JSON):\n{notes_json}") 
    ])

SYNTH_PROMPT = ChatPromptTemplate.from_messages([ ("system", "You are a careful research writer. Use ONLY the provided notes.\n" "Write bullet points and include citations like [1], [2]...\n" "End with a Sources section mapping each [n] to its URL."),
 ("human", "Question: {question}\n\nNotes:\n{notes}") ])

def supervisor_node(state:ResearchState)->ResearchState:
    notes_json = json.dumps(state.notes)
    msg=SUPERVISOR_PROMPT.format_messages(question=state.question, notes_json=notes_json)
    output=llm.invoke(msg)
    try:
        state.next = json.loads(output.content).get("next", "synthesize")
    except Exception as e:
        print(f"Error in supervisor_node: {e}")
        state.next = "synthesize"
    return state

def web_search_node(state:ResearchState)->ResearchState:
    results=ddg.invoke(state.question)
    for item in results[:5]:
        url=item.get("link") or item.get("url")
        snippet=item.get("snippet") or item.get("body") or item.get("text") or ""
        title=item.get("title") or "DuckDuckGo Search Result"
        if snippet:
            state.notes.append({"source":"web","url":url,"title":title,"snippet":snippet})
    state.next=None
    return state

def wiki_research_node(state:ResearchState)->ResearchState:
    text=wiki.invoke(state.question)
    state.notes.append({"source":"wiki","snippet": str(text)[:800], "url": "https://en.wikipedia.org"})
    state.next=None
    return state

def synthesize_node(state: ResearchState) -> ResearchState:
    notes_lines=[
        f"[{i}] {n.get('source','')}: {n.get('snippet','')} (URL: {n.get('url','')})" 
        for i, n in enumerate(state.notes, start=1)
    ]
    msg=SYNTH_PROMPT.format_messages(question=state.question, notes="\n".join(notes_lines))
    answer=llm.invoke(msg).content
    state.messages.append(AIMessage(content=answer))
    state.next="done"
    return state

def route(state: ResearchState) -> str:
    return state.next or "synthesize"

graph=StateGraph(ResearchState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("web_search", web_search_node)
graph.add_node("wiki_research", wiki_research_node)
graph.add_node("synthesize", synthesize_node)
graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route,{"web": "web_search", "wiki": "wiki_research", "synthesize": "synthesize", "done": END})
graph.add_edge("web_search", "supervisor")
graph.add_edge("wiki_research", "supervisor")
graph.add_edge("synthesize", END)


app=graph.compile()

result=app.invoke({"question": "How does the human brain work?"})
print(result["messages"][-1].content)