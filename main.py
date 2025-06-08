import os
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.padding import Padding # Not used yet, but good for rich
import logging
from typing import List, Dict, Tuple, Optional, Union # Added Union

# Load environment variables from .env file
load_dotenv()

# Configure logging BEFORE other imports that might use logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbose output
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        # logging.FileHandler("rag_app.log") # Optionally log to a file
    ]
)
logger = logging.getLogger(__name__)

# Now import project-specific modules
from config import GOOGLE_API_KEY, CLI_HISTORY_LENGTH # To check if key is loaded
from rag_pipeline.graph_builder import rag_app # Your compiled LangGraph app
from rag_pipeline.graph_state import GraphState # The state definition
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage # Added BaseMessage

# Initialize Rich Console
console = Console()

def create_welcome_panel():
    return Panel(
        Text("Welcome to the Civil Engineering RAG CLI!\nAsk me anything about your documents.", justify="center"),
        title="[bold cyan]Civil Engineering RAG System[/bold cyan]",
        border_style="cyan",
        padding=(1, 2)
    )

def build_chat_panel_content(chat_history: List[BaseMessage],
                             current_user_input: Optional[str] = None,
                             ai_thinking: bool = False,
                             ai_response_text: Optional[str] = None,
                             citations_list: Optional[List[Dict]] = None,
                             intermediate_steps: Optional[Text] = None) -> Text: # Added intermediate_steps
    """Builds the Text object for the chat panel."""
    content = Text()
    display_limit = CLI_HISTORY_LENGTH * 2  # user + ai messages

    for msg in chat_history[-display_limit:]:
        if isinstance(msg, HumanMessage):
            content.append("You: ", style="bold green")
            content.append(msg.content + "\n")
        elif isinstance(msg, AIMessage):
            content.append("AI: ", style="bold blue")
            content.append(msg.content + "\n")
            # If this AI message had citations, we'd ideally store them with the message
            # For now, we'll append citations only for the *current* AI response.

    if current_user_input: # Display the query that's being processed
        content.append("You: ", style="bold green")
        content.append(current_user_input + "\n")

    if ai_thinking:
        content.append("AI: ", style="italic blue")
        content.append("Thinking...\n")
        if intermediate_steps: # Display intermediate steps if provided
            content.append(intermediate_steps)


    if ai_response_text: # This is for the final AI response after thinking
        content.append("AI: ", style="bold blue")
        content.append(ai_response_text + "\n")
        if citations_list:
            content.append("\nCitations:\n", style="bold yellow")
            for cit in citations_list:
                source = cit.get('source_file', 'N/A')
                page = cit.get('page_number', 'N/A')
                content.append(f"  - {source}, Page: {page}\n", style="yellow")
    
    return content


def run_cli():
    if not GOOGLE_API_KEY:
        console.print("[bold red]ERROR: GOOGLE_API_KEY is not set in your .env file![/bold red]")
        console.print("Please ensure it's correctly configured.")
        return

    console.print(create_welcome_panel())
    console.print("Type 'exit' or 'quit' to end, 'clear' to reset chat.")

    chat_history: List[BaseMessage] = []

    while True:
        try:
            console.rule(style="dim cyan") 
            if chat_history:
                 console.print(Panel(build_chat_panel_content(chat_history), title="Chat History", border_style="dim cyan", expand=False))

            user_input = Prompt.ask("[bold green]You[/bold green]")

            if user_input.lower() in ['exit', 'quit']:
                console.print("[bold yellow]Exiting RAG CLI. Goodbye![/bold yellow]")
                break
            
            if user_input.lower() == 'clear':
                chat_history = []
                console.clear() 
                console.print(create_welcome_panel())
                console.print("Type 'exit' or 'quit' to end, 'clear' to reset chat.")
                continue

            if not user_input.strip():
                continue

            # --- Prepare for Graph Invocation ---
            current_invocation_chat_history = list(chat_history) 

            initial_state = GraphState(
                original_query=user_input,
                chat_history=current_invocation_chat_history,
                processed_query="",
                target_files_explicit=None,
                inferred_target_files=None,
                retrieval_filter=None,
                retrieved_docs=[],
                query_requires_doc_search=False,
                doc_search_performed=False,
                use_external_knowledge_explicitly_forbidden=False,
                should_compare_with_general_knowledge=False,
                answer_from_docs_only=None,
                answer_from_general_knowledge=None,
                synthesized_answer=None,
                final_response_for_user="",
                citations=[],
                error_message=None
            )

            logger.info(f"Invoking RAG graph with query: '{user_input}'")
            
            final_state_result = None
            ai_response_content = "Sorry, I couldn't process that." 
            citations_list = []
            intermediate_steps_display = Text() # For accumulating node execution messages

            # --- Streaming with Live ---
            with Live(console=console, refresh_per_second=4, auto_refresh=True, transient=False) as live_session:
                # Initial "Thinking" display
                live_session.update(Panel(build_chat_panel_content(chat_history, 
                                                                   current_user_input=user_input, 
                                                                   ai_thinking=True,
                                                                   intermediate_steps=intermediate_steps_display), 
                                          title="Chat Log", border_style="dim cyan", expand=False))
                
                try:
                    # Corrected stream iteration:
                    for stream_event_dict in rag_app.stream(
                        initial_state,
                        config={"recursion_limit": 25}, # Adjust recursion_limit as needed
                        stream_mode="updates" 
                    ):
                        if not stream_event_dict: 
                            continue

                        event_type = list(stream_event_dict.keys())[0] # Node name
                        event_data = stream_event_dict[event_type]   # GraphState after this node

                        logger.debug(f"Stream Event: Node '{event_type}' completed.")
                        
                        # Update intermediate steps display
                        intermediate_steps_display.append(f"  Executed: {event_type}...\n", style="italic dim blue")
                        live_session.update(Panel(build_chat_panel_content(chat_history, 
                                                                           current_user_input=user_input, 
                                                                           ai_thinking=True,
                                                                           intermediate_steps=intermediate_steps_display), 
                                                  title="Chat Log", border_style="dim cyan", expand=False))

                        final_state_result = event_data # Update with the latest state

                    # After the loop, final_state_result holds the state from the last node
                    if final_state_result:
                        ai_response_content = final_state_result.get("final_response_for_user", "Error: Could not extract final response.")
                        citations_list = final_state_result.get("citations", [])
                        if final_state_result.get("error_message") and not ai_response_content.startswith("An error occurred"):
                            # Ensure error message from state is shown if final_response_for_user wasn't updated by error_handler
                            ai_response_content = f"An error occurred: {final_state_result.get('error_message')}"
                    else:
                        logger.error("RAG app stream did not yield a final state result (final_state_result is None after loop).")
                        ai_response_content = "Error: The RAG process did not complete as expected."

                except Exception as stream_ex:
                    logger.error(f"Error during RAG app stream: {stream_ex}", exc_info=True)
                    ai_response_content = f"An error occurred during processing: {stream_ex}"
                    citations_list = [] # Clear citations on stream error
                
                # Final update to Live session to show the complete response
                live_session.update(Panel(build_chat_panel_content(chat_history, 
                                                                   current_user_input=user_input, # Keep showing the query it responded to
                                                                   ai_thinking=False, # No longer thinking
                                                                   ai_response_text=ai_response_content, 
                                                                   citations_list=citations_list,
                                                                   intermediate_steps=None), # Clear intermediate steps
                                           title="Chat Log", border_style="dim cyan", expand=False))

            # --- Update actual chat_history outside the Live context for the next iteration ---
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=ai_response_content)) # Store the AI response

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Exiting due to user interrupt.[/bold yellow]")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in the CLI loop: {e}", exc_info=True)
            console.print(f"[bold red]Critical Error: {e}[/bold red]")
            # Add to history so user sees the error in context
            current_input_for_error = user_input if 'user_input' in locals() else "Previous action"
            chat_history.append(HumanMessage(content=current_input_for_error))
            chat_history.append(AIMessage(content=f"Sorry, a critical error occurred: {e}"))

if __name__ == "__main__":
    run_cli()