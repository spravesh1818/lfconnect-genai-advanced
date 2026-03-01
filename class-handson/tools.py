from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


wiki = WikipediaQueryRun(

api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1200)

)

ddg = DuckDuckGoSearchResults(max_results=5, output_format="list")