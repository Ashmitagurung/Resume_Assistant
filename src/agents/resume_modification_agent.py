# src/agents/resume_modification_agent.py
# Resume Modification Tool-Calling Agent

import json
import streamlit as st
from typing import Dict, List, Any
from langchain_core.tools import Tool
from langchain.agents import initialize_agent, AgentType


class ResumeModificationAgent:
    """Agent that uses tools to suggest resume modifications"""

    def __init__(self, llm, vectorstore, qa_chain):
        self.llm = llm
        self.vectorstore = vectorstore
        self.qa_chain = qa_chain
        self.tools = self._create_tools()
        self.agent = self._initialize_agent()

    def _create_tools(self):
        """Create tools for resume modification suggestions"""

        def analyze_skills_gap(role_and_target: str) -> str:
            """Analyze skills gap between current resume and target position"""
            try:
                parts = role_and_target.split(" target: ")
                if len(parts) != 2:
                    return "Please provide input in format: 'current_role target: target_position'"

                current_role, target_position = parts

                # Get current skills
                current_skills_result = self.qa_chain({
                    "query": f"What skills does the {current_role} have?"
                })

                # Analyze gap
                gap_analysis_result = self.qa_chain({
                    "query": f"What skills would be needed for {target_position} that might be missing from current {current_role} resume?"
                })

                response = f"Skills Gap Analysis:\n\n"
                response += f"Current Skills ({current_role}):\n{current_skills_result.get('result', 'No skills found')}\n\n"
                response += f"Gap Analysis for {target_position}:\n{gap_analysis_result.get('result', 'No gap analysis available')}\n"

                return response
            except Exception as e:
                return f"Error analyzing skills gap: {str(e)}"

        def suggest_experience_improvements(role_and_section: str) -> str:
            """Suggest improvements for experience section"""
            try:
                parts = role_and_section.split(" section: ")
                if len(parts) != 2:
                    return "Please provide input in format: 'role_name section: experience_details'"

                role, experience_context = parts

                # Get current experience
                current_exp_result = self.qa_chain({
                    "query": f"What work experience does the {role} have?"
                })

                # Generate improvement suggestions
                improvement_query = f"""
                Based on the current experience of {role}: {current_exp_result.get('result', '')},
                suggest specific improvements for the experience section considering: {experience_context}
                Focus on:
                1. Quantifying achievements with numbers/metrics
                2. Using stronger action verbs
                3. Highlighting relevant accomplishments
                4. Better formatting and presentation
                """

                improvement_result = self.qa_chain({"query": improvement_query})

                response = f"Experience Section Improvement Suggestions:\n\n"
                response += f"Current Experience:\n{current_exp_result.get('result', 'No experience found')}\n\n"
                response += f"Suggested Improvements:\n{improvement_result.get('result', 'No suggestions available')}\n"

                return response
            except Exception as e:
                return f"Error suggesting experience improvements: {str(e)}"

        def optimize_resume_format(role_and_target: str) -> str:
            """Suggest formatting and structure optimizations"""
            try:
                parts = role_and_target.split(" for: ")
                if len(parts) != 2:
                    return "Please provide input in format: 'role_name for: target_industry/position'"

                role, target = parts

                # Analyze current structure
                structure_result = self.qa_chain({
                    "query": f"Analyze the overall structure and organization of the {role} resume"
                })

                response = f"Resume Format Optimization for {target}:\n\n"
                response += f"Current Structure Analysis:\n{structure_result.get('result', 'No structure analysis available')}\n\n"

                # General formatting suggestions
                response += "Recommended Formatting Improvements:\n"
                response += "1. Use consistent bullet points and formatting throughout\n"
                response += "2. Ensure proper section ordering: Contact → Summary → Experience → Skills → Education\n"
                response += "3. Use action verbs at the beginning of bullet points\n"
                response += "4. Quantify achievements with specific numbers and metrics\n"
                response += "5. Keep each bullet point to 1-2 lines maximum\n"
                response += "6. Use consistent date formatting (MM/YYYY)\n"
                response += "7. Ensure adequate white space for readability\n"
                response += f"8. Tailor content specifically for {target} industry standards\n"

                return response
            except Exception as e:
                return f"Error optimizing resume format: {str(e)}"

        def suggest_keyword_optimization(role_and_job_desc: str) -> str:
            """Suggest keyword optimizations for ATS systems"""
            try:
                parts = role_and_job_desc.split(" keywords: ")
                if len(parts) != 2:
                    return "Please provide input in format: 'role_name keywords: target_job_keywords'"

                role, target_keywords = parts

                # Get current resume content
                current_content_result = self.qa_chain({
                    "query": f"What are the main keywords and terms used in the {role} resume?"
                })

                response = f"Keyword Optimization Suggestions:\n\n"
                response += f"Current Keywords in Resume:\n{current_content_result.get('result', 'No keywords found')}\n\n"
                response += f"Target Keywords to Include: {target_keywords}\n\n"

                # Generate keyword suggestions
                keyword_analysis = f"""
                Suggest how to naturally incorporate these target keywords: {target_keywords}
                into the {role} resume without keyword stuffing. Focus on:
                1. Skills section optimization
                2. Experience description enhancements
                3. Summary/objective improvements
                4. Technical proficiencies alignment
                """

                keyword_result = self.qa_chain({"query": keyword_analysis})
                response += f"Keyword Integration Strategy:\n{keyword_result.get('result', 'No strategy available')}\n"

                return response
            except Exception as e:
                return f"Error suggesting keyword optimization: {str(e)}"

        def generate_tailored_summary(role_and_target: str) -> str:
            """Generate a tailored professional summary"""
            try:
                parts = role_and_target.split(" for: ")
                if len(parts) != 2:
                    return "Please provide input in format: 'role_name for: target_position'"

                role, target_position = parts

                # Get candidate's background
                background_result = self.qa_chain({
                    "query": f"Summarize the key qualifications, skills, and experience of the {role}"
                })

                # Generate tailored summary
                summary_query = f"""
                Create a compelling professional summary for a {role} targeting a {target_position} role.
                Based on their background: {background_result.get('result', '')}

                The summary should:
                1. Be 3-4 sentences long
                2. Highlight most relevant experience and skills
                3. Include quantifiable achievements if available
                4. Match the tone and requirements of {target_position}
                5. Use industry-specific keywords
                """

                summary_result = self.qa_chain({"query": summary_query})

                response = f"Tailored Professional Summary for {target_position}:\n\n"
                response += f"Current Background:\n{background_result.get('result', 'No background found')}\n\n"
                response += f"Suggested Professional Summary:\n{summary_result.get('result', 'No summary generated')}\n"

                return response
            except Exception as e:
                return f"Error generating tailored summary: {str(e)}"

        # Create Tool objects
        tools = [
            Tool(
                name="analyze_skills_gap",
                description="Analyze skills gap between current resume and target position. Input format: 'current_role target: target_position'",
                func=analyze_skills_gap
            ),
            Tool(
                name="suggest_experience_improvements",
                description="Suggest improvements for experience section. Input format: 'role_name section: experience_details'",
                func=suggest_experience_improvements
            ),
            Tool(
                name="optimize_resume_format",
                description="Suggest formatting and structure optimizations. Input format: 'role_name for: target_industry/position'",
                func=optimize_resume_format
            ),
            Tool(
                name="suggest_keyword_optimization",
                description="Suggest keyword optimizations for ATS systems. Input format: 'role_name keywords: target_job_keywords'",
                func=suggest_keyword_optimization
            ),
            Tool(
                name="generate_tailored_summary",
                description="Generate a tailored professional summary. Input format: 'role_name for: target_position'",
                func=generate_tailored_summary
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
            st.error(f"Error initializing modification agent: {str(e)}")
            return None

    def suggest_modifications(self, query: str) -> str:
        """Main method to suggest resume modifications using the agent"""
        if not self.agent:
            return "Error: Agent not initialized properly"

        try:
            with st.spinner("AI Agent is analyzing resume for improvements..."):
                # Enhanced prompt for better tool selection
                enhanced_query = f"""
                Analyze this resume modification request and use the appropriate tools to provide comprehensive suggestions:

                Request: {query}

                Please:
                1. Determine which tool(s) would be most helpful for this modification request
                2. Use the tools to analyze the current resume
                3. Provide specific, actionable improvement suggestions
                4. Include examples where possible
                5. Prioritize suggestions based on impact
                """

                result = self.agent.run(enhanced_query)
                return result

        except Exception as e:
            return f"Error during resume modification analysis: {str(e)}"

    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return [tool.name for tool in self.tools]