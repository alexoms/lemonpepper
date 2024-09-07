from langchain_milvus_rag_chat_api import RAGSystem
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.tools import ShellTool
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Dict, Any
import re
import subprocess
import os

class ExtendedRAGSystem:
    def __init__(self, ollama_server: str, milvus_host: str, milvus_port: str, working_dir: str = None):
        self.rag_system = RAGSystem(ollama_server, milvus_host, milvus_port)
        self.llm = Ollama(
            model="llama3.1:latest",
            base_url=ollama_server,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
        self.shell_tool = ShellTool()
        self.shell_tool.description = self.shell_tool.description + f"args {self.shell_tool.args}".replace(
            "{", "{{"
        ).replace("}", "}}")
        self.agent = initialize_agent(
            [self.shell_tool],
            self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        self.shell_command_chain = self.create_shell_command_chain()
        self.working_dir = working_dir if working_dir else os.getcwd()

    def create_shell_command_chain(self):
        prompt = PromptTemplate(
            input_variables=["query", "response", "working_dir"],
            template="""
            Given the following query and response:
            Query: {query}
            Response: {response}
            Working Directory: {working_dir}

            Determine if a shell command is needed to complete the task described in the response.
            If a shell command is needed, formulate the appropriate command.
            If no shell command is needed, respond with "No shell command needed."

            Your response should be in the format:
            Shell Command Needed: [Yes/No]
            Command: [The shell command, if needed]

            Remember to be cautious with potentially dangerous commands and always prioritize safety.
            Use the provided working directory when formulating file paths in the command.
            """
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def process_document(self, file_path: str) -> None:
        self.rag_system.process_document(file_path)

    def create_file(self, file_name: str, content: str) -> str:
        try:
            file_path = os.path.join(self.working_dir, file_name)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"File '{file_name}' created successfully in {self.working_dir}"
        except Exception as e:
            return f"Error creating file: {str(e)}"

    def run_shell_command(self, command: str) -> str:
        try:
            if command.strip().lower().startswith("echo") and ">" in command:
                # Extract file name and content from the echo command
                parts = command.split(">", 1)
                content = parts[0].replace("echo", "", 1).strip().strip('"').strip("'")
                file_name = parts[1].strip().strip('"').strip("'")
                return self.create_file(file_name, content)
            else:
                # For other commands, use subprocess as before
                result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, cwd=self.working_dir)
                return f"Command executed in {self.working_dir}\nResult: {result.stdout}"
        except subprocess.CalledProcessError as e:
            return f"Error executing command in {self.working_dir}: {e.stderr}"
        except Exception as e:
            return f"Error: {str(e)}"

    def query(self, query: str) -> Dict[str, Any]:
        rag_result = self.rag_system.query(query)
        
        shell_decision = self.shell_command_chain.run(query=query, response=rag_result['answer'], working_dir=self.working_dir)
        
        if "Shell Command Needed: Yes" in shell_decision:
            command_match = re.search(r"Command: (.+)", shell_decision, re.DOTALL)
            if command_match:
                command = command_match.group(1).strip()
                try:
                    shell_result = self.run_shell_command(command)
                    return {
                        "answer": f"{rag_result['answer']}\n\nExecuted shell command: {command}\nWorking Directory: {self.working_dir}\nResult: {shell_result}",
                        "source": "rag_and_shell"
                    }
                except Exception as e:
                    return {
                        "answer": f"{rag_result['answer']}\n\nAttempted to execute shell command: {command}\nWorking Directory: {self.working_dir}\nError: {str(e)}",
                        "source": "rag_and_shell_error"
                    }
            else:
                return {
                    "answer": f"{rag_result['answer']}\n\nA shell command was deemed necessary but couldn't be formulated.",
                    "source": "rag_and_shell_error"
                }
        else:
            return rag_result



    def set_working_directory(self, new_working_dir: str):
        if os.path.isdir(new_working_dir):
            self.working_dir = new_working_dir
            return f"Working directory set to: {self.working_dir}"
        else:
            return f"Error: {new_working_dir} is not a valid directory"

    def cleanup(self) -> None:
        self.rag_system.cleanup()

def main():
    ollama_server = "http://192.168.1.81:11434"
    milvus_host = "192.168.1.81"
    milvus_port = "19530"
    
    # Use the current directory as the working directory
    working_dir = os.getcwd()
    
    extended_system = ExtendedRAGSystem(ollama_server, milvus_host, milvus_port, working_dir)
    
    file_path = "http://192.168.1.81:5000/inc/alex_chang_resume_202409.pdf"
    print(f"Processing document: {file_path}")
    extended_system.process_document(file_path)

    print(f"\nCurrent working directory for shell commands: {extended_system.working_dir}")

    while True:
        query = input("\nQuery (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        elif query.lower().startswith('set working directory:'):
            new_dir = query.split(':', 1)[1].strip()
            print(extended_system.set_working_directory(new_dir))
            continue

        result = extended_system.query(query)
        print("\nAnswer:", result['answer'])
        if result['source'] == 'retrieval':
            print("\nSource Documents:")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"Document {i}:")
                print(f"Content: {doc.page_content[:100]}...")
                print(f"Metadata: {doc.metadata}")
                print()
        elif result['source'] in ['rag_and_shell', 'rag_and_shell_error']:
            print("\nShell command was executed as part of the response.")

    extended_system.cleanup()

if __name__ == "__main__":
    main()