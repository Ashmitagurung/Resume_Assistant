# src/agents/resume_search_agent.py
# Resume Search Tool-Calling Agent

import json
import streamlit as st
from typing import Dict, List, Any
from langchain_core.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_core.prompts import PromptTemplate


class ResumeSearchAgent:
    """Agent that uses tools to search and analyze resumes"""

    def __init__(self, llm, vectorstore, qa_chain):
        self.llm = llm
        self.vectorstore = vectorstore
        self.qa_chain = qa_chain
        self.tools = self._create_tools()
        self.agent = self._initialize_agent()

    def _create_tools(self):
        """Create tools for resume searching"""

        def search_by_skills(query: str) -> str:
            """Search for candidates based on specific skills"""
            try:
                search_query = f"skills experience {query}"
                result = self.qa_chain({"query": search_query})

                # Format the response
                answer = result.get("result", "No information found")
                sources = result.get("source_documents", [])

                response = f"Skills Search Results:\n{answer}\n\n"
                if sources:
                    response += "Sources:\n"
                    for i, doc in enumerate(sources[:3]):
                        filename = doc.metadata.get('filename', 'unknown')
                        role = doc.metadata.get('role', 'Unknown Role')
                        response += f"- {filename} ({role})\n"

                return response
            except Exception as e:
                return f"Error searching by skills: {str(e)}"

        def search_by_experience(query: str) -> str:
            """Search for candidates based on work experience"""
            try:
                search_query = f"work experience job position {query}"
                result = self.qa_chain({"query": search_query})

                answer = result.get("result", "No information found")
                sources = result.get("source_documents", [])

                response = f"Experience Search Results:\n{answer}\n\n"
                if sources:
                    response += "Sources:\n"
                    for i, doc in enumerate(sources[:3]):
                        filename = doc.metadata.get('filename', 'unknown')
                        role = doc.metadata.get('role', 'Unknown Role')
                        response += f"- {filename} ({role})\n"

                return response
            except Exception as e:
                return f"Error searching by experience: {str(e)}"

        def search_by_education(query: str) -> str:
            """Search for candidates based on educational background"""
            try:
                search_query = f"education degree university college {query}"
                result = self.qa_chain({"query": search_query})

                answer = result.get("result", "No information found")
                sources = result.get("source_documents", [])

                response = f"Education Search Results:\n{answer}\n\n"
                if sources:
                    response += "Sources:\n"
                    for i, doc in enumerate(sources[:3]):
                        filename = doc.metadata.get('filename', 'unknown')
                        role = doc.metadata.get('role', 'Unknown Role')
                        response += f"- {filename} ({role})\n"

                return response
            except Exception as e:
                return f"Error searching by education: {str(e)}"

        def compare_candidates(query: str) -> str:
            """Compare multiple candidates based on specific criteria"""
            try:
                search_query = f"compare candidates {query}"
                result = self.qa_chain({"query": search_query})

                answer = result.get("result", "No comparison available")
                sources = result.get("source_documents", [])

                response = f"Candidate Comparison:\n{answer}\n\n"
                if sources:
                    response += "Candidates Compared:\n"
                    filenames = set()
                    for doc in sources:
                        filename = doc.metadata.get('filename', 'unknown')
                        role = doc.metadata.get('role', 'Unknown Role')
                        filenames.add(f"{filename} ({role})")

                    for filename in filenames:
                        response += f"- {filename}\n"

                return response
            except Exception as e:
                return f"Error comparing candidates: {str(e)}"

        def get_role_summary(role: str) -> str:
            """Get summary of all candidates for a specific role"""
            try:
                search_query = f"summary overview {role} candidates"
                result = self.qa_chain({"query": search_query})

                answer = result.get("result", "No summary available")
                sources = result.get("source_documents", [])

                # Count candidates for this role
                role_candidates = [doc for doc in sources if doc.metadata.get('role') == role]

                response = f"Role Summary for {role}:\n{answer}\n\n"
                response += f"Number of {role} candidates found: {len(role_candidates)}\n"

                return response
            except Exception as e:
                return f"Error getting role summary: {str(e)}"

        # Create Tool objects
        tools = [
            Tool(
                name="search_by_skills",
                description="Search for candidates based on specific skills. Input should be skill names or technologies.",
                func=search_by_skills
            ),
            Tool(
                name="search_by_experience",
                description="Search for candidates based on work experience. Input should be job titles, companies, or experience requirements.",
                func=search_by_experience
            ),
            Tool(
                name="search_by_education",
                description="Search for candidates based on educational background. Input should be degrees, universities, or educational requirements.",
                func=search_by_education
            ),
            Tool(
                name="compare_candidates",
                description="Compare multiple candidates based on specific criteria. Input should be comparison criteria.",
                func=compare_candidates
            ),
            Tool(
                name="get_role_summary",
                description="Get a summary of all candidates for a specific role. Input should be the role name.",
                func=get_role_summary
            )
        ]

        return tools

    def _initialize_agent(self):
        """Initialize the tool-calling agent"""
        try:
            agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3
            )
            return agent
        except Exception as e:
            st.error(f"Error initializing search agent: {str(e)}")
            return None

    def search_resumes(self, query: str) -> str:
        """Main method to search resumes using the agent"""
        if not self.agent:
            return "Error: Agent not initialized properly"

        try:
            with st.spinner("AI Agent is analyzing resumes..."):
                # Enhanced prompt for better tool selection
                enhanced_query = f"""
                Analyze this resume search request and use the appropriate tools to provide a comprehensive answer:

                Query: {query}

                Please:
                1. Determine which tool(s) would be most helpful for this query
                2. Use the tools to gather information
                3. Provide a clear, organized response
                4. Include specific examples and details from the resumes when possible
                """

                result = self.agent.run(enhanced_query)
                return result

        except Exception as e:
            return f"Error during resume search: {str(e)}"

    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return [tool.name for tool in self.tools]