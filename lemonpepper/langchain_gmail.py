from langchain_milvus_rag_chat_api import RAGSystem
from langchain.agents import AgentType, initialize_agent, Tool
from langchain_community.tools import ShellTool
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Dict, Any, List
import re
import subprocess
import os
import pickle
import base64
import imaplib
import email
from email.header import decode_header
import json

class ExtendedRAGSystem:
    def __init__(self, ollama_server: str, milvus_host: str, milvus_port: str, working_dir: str = None, gmail_address: str = None, gmail_password: str = None):
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
        self.gmail_address = gmail_address
        self.gmail_password = gmail_password
        self.imap_server = "imap.gmail.com"
        self.resume_content = self.get_resume_content()
        self.job_analysis_chain = self.create_job_analysis_chain()
        
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

    def handle_file_creation(self, query: str, content: str) -> str:
        file_name_match = re.search(r'save.*as.*?(\w+\.\w+)', query, re.IGNORECASE)
        if file_name_match:
            file_name = file_name_match.group(1)
            code_content = self.extract_code(content)
            if code_content:
                result = self.create_file(file_name, code_content)
                return f"File creation result: {result}"
            else:
                return "Could not extract Python code from the response."
        else:
            return "Could not determine the file name from the query."

    def extract_code(self, content: str) -> str:
        # Extract code between triple backticks
        code_blocks = re.findall(r'```python(.*?)```', content, re.DOTALL)
        if code_blocks:
            # Return the last code block (in case there are multiple)
            code = code_blocks[-1].strip()
        else:
            # If no code blocks found, try to extract based on indentation
            lines = content.split('\n')
            code_lines = [line for line in lines if line.strip().startswith('def ') or line.strip().startswith('import ') or '    ' in line]
            code = '\n'.join(code_lines)
        
        # Remove comments about saving the script
        code_lines = [line for line in code.split('\n') if not line.strip().startswith('# Save this script')]
        
        # Remove any trailing shell prompt characters or other non-code content
        code = '\n'.join(code_lines)
        code = re.sub(r'[\s%#>]+$', '', code, flags=re.MULTILINE)
        
        # Ensure no trailing whitespace or newlines
        return code.rstrip()

    def create_file(self, file_name: str, content: str) -> str:
        try:
            file_path = os.path.join(self.working_dir, file_name)
            
            # Clean the content
            cleaned_content = self.clean_content(content)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            return f"File '{file_name}' created successfully in {self.working_dir}"
        except Exception as e:
            return f"Error creating file: {str(e)}"

    def clean_content(self, content: str) -> str:
        # Remove any trailing whitespace, newlines, or shell artifacts
        cleaned = re.sub(r'\s*%.*$', '', content, flags=re.MULTILINE)
        cleaned = cleaned.rstrip()
        
        # Ensure the content ends with a single newline
        return cleaned + '\n'

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
        
        # Check if the query involves file creation
        if "save" in query.lower() and "file" in query.lower():
            file_creation_result = self.handle_file_creation(query, rag_result['answer'])
            return {
                "answer": f"{rag_result['answer']}\n\n{file_creation_result}",
                "source": "rag_and_file_creation"
            }
        
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

    def connect_to_gmail(self):
        mail = imaplib.IMAP4_SSL(self.imap_server)
        mail.login(self.gmail_address, self.gmail_password)
        return mail

    def get_emails(self, query: str = "ALL", max_results: int = 10) -> List[Dict[str, Any]]:
        mail = self.connect_to_gmail()
        mail.select('inbox')
        _, search_data = mail.search(None, query)
        email_ids = search_data[0].split()[-max_results:]
        emails = []
        for email_id in email_ids:
            _, msg_data = mail.fetch(email_id, '(RFC822)')
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    email_body = response_part[1]
                    email_message = email.message_from_bytes(email_body)
                    subject = decode_header(email_message["Subject"])[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode()
                    emails.append({"id": email_id, "message": email_message, "subject": subject})
        mail.close()
        mail.logout()
        return emails

    def get_email_content(self, email_message: email.message.Message) -> str:
        def decode_content(part):
            content = part.get_payload(decode=True)
            charset = part.get_content_charset()
            if charset is None:
                # Try common encodings
                for encoding in ['utf-8', 'iso-8859-1', 'windows-1252']:
                    try:
                        return content.decode(encoding)
                    except UnicodeDecodeError:
                        continue
                # If all else fails, use 'replace' to handle decoding errors
                return content.decode('utf-8', errors='replace')
            else:
                return content.decode(charset)

        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    return decode_content(part)
        else:
            return decode_content(email_message)
        
        # If no text content is found, return an empty string
        return ""

    def analyze_job_email(self, email_data: Dict[str, Any], resume_summary: str) -> Dict[str, Any]:
        content = self.get_email_content(email_data["message"])
        analysis = self.job_analysis_chain.run(resume_summary=resume_summary, subject=email_data["subject"], content=content)
        return self.parse_job_analysis(analysis, email_data["subject"])

    def parse_job_analysis(self, analysis: str, subject: str) -> Dict[str, Any]:
        lines = analysis.strip().split('\n')
        is_job_posting = 'yes' in lines[0].lower()
        is_remote = 'yes' in lines[1].lower()
        matches_resume = 'yes' in lines[2].lower()
        explanation = ' '.join(lines[3:])
        return {
            'subject': subject,
            'is_job_posting': is_job_posting,
            'is_remote': is_remote,
            'matches_resume': matches_resume,
            'explanation': explanation
        }

    def get_resume_content(self) -> str:
        query = "Retrieve the full content of Alex Chang's resume"
        docs = self.rag_system.system["vector_store"].similarity_search(query, k=10)  # Retrieve top 10 most relevant chunks
        print(f"Retrieved {len(docs)} chunks from the vector store.")
        
        # Filter for chunks from Alex Chang's resume and sort by chunk_id
        resume_chunks = sorted(
            [doc for doc in docs if "alex_chang_resume" in doc.metadata['source']],
            key=lambda x: x.metadata['chunk_id']
        )
        print(f"Filtered {len(resume_chunks)} chunks related to Alex Chang's resume.")
        
        if not resume_chunks:
            print("No resume chunks found. Debugging information:")
            for doc in docs:
                print(f"Chunk source: {doc.metadata.get('source')}")
                print(f"Chunk content preview: {doc.page_content[:100]}...")
                print("---")
        
        # Combine chunks into a single string
        resume_content = "\n\n".join([chunk.page_content for chunk in resume_chunks])
        
        print(f"Final resume content length: {len(resume_content)} characters")
        return resume_content
    
    def create_chain(self, prompt):
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def create_job_analysis_chain(self):
        template = """
        Analyze the following email to determine if it's a job posting, if it's for a remote position, and if it matches the provided resume content.

        Resume Content:
        {resume_content}

        Email Subject: {subject}
        Email Content: {content}

        Provide your analysis in the following JSON format:
        {{
            "is_job_posting": boolean,
            "is_remote": boolean or null if unclear (hybrid is False and onsite is False),
            "matches_resume": boolean,
            "explanation": "string explaining your decisions (3-4 sentences)"
        }}

        Analysis:
        """

        prompt = PromptTemplate(
            input_variables=["resume_content", "subject", "content"],
            template=template
        )

        return self.create_chain(prompt)

    def analyze_job_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        content = self.get_email_content(email_data["message"])
        analysis = self.job_analysis_chain.run(resume_content=self.resume_content, subject=email_data["subject"], content=content)
        return self.parse_job_analysis(analysis, email_data["subject"])

    def parse_job_analysis(self, analysis: str, subject: str) -> Dict[str, Any]:
        try:
            # Extract the JSON part from the analysis
            json_start = analysis.find('{')
            json_end = analysis.rfind('}') + 1
            json_str = analysis[json_start:json_end]
            
            # Parse the JSON
            result = json.loads(json_str)
            
            # Add the subject to the result
            result['subject'] = subject
            
            return result
        except json.JSONDecodeError:
            # If JSON parsing fails, return a default structure
            return {
                'subject': subject,
                'is_job_posting': False,
                'is_remote': None,
                'matches_resume': False,
                'explanation': "Failed to parse the analysis output."
            }

    def analyze_job_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        content = self.get_email_content(email_data["message"])
        analysis = self.job_analysis_chain.run(resume_content=self.resume_content, subject=email_data["subject"], content=content)
        return self.parse_job_analysis(analysis, email_data["subject"])

    def analyze_gmail_jobs(self, query: str = "ALL", max_results: int = 10) -> List[Dict[str, Any]]:
        emails = self.get_emails(query, max_results)
        results = []
        job_related_count = 0
        
        print(f"\nAnalyzing {len(emails)} emails...")
        
        for i, email_data in enumerate(emails, 1):
            print(f"Processing email {i}/{len(emails)}...")
            try:
                analysis = self.analyze_job_email(email_data)
                if analysis['is_job_posting']:
                    job_related_count += 1
                    results.append(analysis)
                    print(f"\nJob Posting #{job_related_count}:")
                    print(f"Subject: {analysis['subject']}")
                    print(f"Is Remote: {analysis['is_remote']}")
                    print(f"Matches Resume: {analysis['matches_resume']}")
                    print(f"Explanation: {analysis['explanation']}")
                    print("-" * 50)
            except Exception as e:
                print(f"Error processing email {i}: {str(e)}")
        
        print(f"\nAnalysis Summary:")
        print(f"Total emails analyzed: {len(emails)}")
        print(f"Job-related emails found: {job_related_count}")
        print(f"Potential job matches: {len([r for r in results if r['matches_resume']])}")
        
        return results

# Example usage
def main():
    ollama_server = "http://192.168.1.81:11434"
    milvus_host = "192.168.1.81"
    milvus_port = "19530"
    working_dir = "/path/to/your/working/directory"
    gmail_address = "alexchang@alumni.ucla.edu"
    gmail_password = ""  # Use an app-specific password
    
    extended_system = ExtendedRAGSystem(ollama_server, milvus_host, milvus_port, working_dir, gmail_address, gmail_password)
    
    file_path = "http://192.168.1.81:5000/inc/alex_chang_resume_202409.pdf"
    print(f"Processing document: {file_path}")
    extended_system.process_document(file_path)
    print("\nRetrieving Resume Content:")
    resume_content = extended_system.get_resume_content()
    print(f"Resume Content Length: {len(resume_content)}")
    print("Resume Content:")
    print(extended_system.resume_content[:50000] + "...")  # Print first 500 characters
    print("\nAnalyzing job emails...")

    job_results = extended_system.analyze_gmail_jobs(query="ALL", max_results=20)
    
    for job in job_results:
        print(f"\nSubject: {job['subject']}")
        print(f"Is Remote: {job['is_remote']}")
        print(f"Matches Resume: {job['matches_resume']}")
        print(f"Explanation: {job['explanation']}")

if __name__ == "__main__":
    main()