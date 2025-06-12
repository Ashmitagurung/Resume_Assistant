# Tab renderer functions for the main interface
# Contains functions to render each of the Streamlit tabs

import streamlit as st
import re


def extract_person_name_from_query(query):
    """Extract person name from query if mentioned"""
    # Common patterns for person names in queries
    query_lower = query.lower()

    # Look for patterns like "abin prajapati", "prasanna ghimire", etc.
    # This is a simple approach - you might want to enhance this based on your specific name patterns
    name_patterns = [
        r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b',  # First Last name pattern
        r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b'  # Another pattern for names
    ]

    for pattern in name_patterns:
        matches = re.findall(pattern, query)
        if matches:
            if isinstance(matches[0], tuple):
                return ' '.join(matches[0])
            else:
                return matches[0]

    return None


def filter_sources_by_context(sources, query, person_name=None):
    """Filter sources based on query context and person name"""
    if not person_name:
        person_name = extract_person_name_from_query(query)

    if person_name:
        # Filter sources to only include those from the mentioned person
        filtered_sources = []
        for doc in sources:
            filename = doc.metadata.get('filename', '').lower()
            # Check if person name is in filename
            if any(name_part.lower() in filename for name_part in person_name.split()):
                filtered_sources.append(doc)

        # If we found relevant sources, return them; otherwise return original sources
        if filtered_sources:
            return filtered_sources

    return sources


def render_query_tab(initialized, qa_chain, query_rag):
    """Render the query tab content"""
    st.header("Ask about the resumes")

    # Query input
    query = st.text_area("Enter your question:",
                         placeholder="Example: What skills do the candidates have?",
                         disabled=not initialized)

    query_button = st.button("Ask", disabled=not (initialized and query))

    # Display results for query
    if query_button and query:
        result = query_rag(qa_chain, query)

        st.subheader("Answer:")
        st.write(result["result"])

        # Extract person name from query for better source filtering
        person_name = extract_person_name_from_query(query)

        # Filter sources based on context
        filtered_sources = filter_sources_by_context(result["source_documents"], query, person_name)

        # Limit to 3 most relevant sources
        relevant_sources = filtered_sources[:3]

        st.subheader("Sources:")

        if not relevant_sources:
            st.info("No specific sources found for this query.")
        else:
            for i, doc in enumerate(relevant_sources):
                filename = doc.metadata.get('filename', 'unknown')
                role = doc.metadata.get('role', 'Unknown Role')

                with st.expander(f"Source {i + 1} from {filename} ({role})"):
                    content = doc.page_content
                    # Show preview of content
                    preview = content[:500] + "..." if len(content) > 500 else content
                    st.write(preview)


def render_role_tab(initialized, all_roles, vectorstore, qa_chain, get_resume_by_role, query_rag):
    """Render the role-based search tab content"""
    st.header("Search by Role")

    if initialized and all_roles:
        selected_role = st.selectbox("Select a role:", all_roles)

        search_role_button = st.button("View Resume", key="search_role_button")

        if search_role_button and selected_role:
            with st.spinner(f"Retrieving resumes for role: {selected_role}"):
                resumes = get_resume_by_role(vectorstore, selected_role)

                if not resumes:
                    st.warning(f"No resumes found for role: {selected_role}")
                else:
                    for filename, info in resumes.items():
                        st.subheader(f"Resume: {filename}")
                        st.write(f"**Role:** {info['role']}")

                        # Combine the content chunks and show a preview
                        full_content = " ".join(info['content'])
                        with st.expander("Resume Content Preview"):
                            st.write(full_content[:1000] + "..." if len(full_content) > 1000 else full_content)

                        # Run a few preset queries about this resume
                        with st.spinner("Analyzing resume details..."):
                            # Make queries more specific to avoid cross-contamination
                            skills_query = f"What specific skills does the person in {filename} have? Only answer based on {filename}."
                            experience_query = f"What is the work experience of the person in {filename}? Only answer based on {filename}."
                            education_query = f"What is the education background of the person in {filename}? Only answer based on {filename}."

                            skills_result = query_rag(qa_chain, skills_query)
                            experience_result = query_rag(qa_chain, experience_query)
                            education_result = query_rag(qa_chain, education_query)

                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("##### Skills")
                                st.write(skills_result["result"])

                                st.markdown("##### Education")
                                st.write(education_result["result"])

                            with col2:
                                st.markdown("##### Experience")
                                st.write(experience_result["result"])
    else:
        st.info("Please upload and process resumes first to view roles.")


def render_modification_tab(initialized, all_roles, qa_chain, vectorstore, get_resume_by_role, query_rag):
    """Render the resume modification suggestions tab content"""
    st.header("Resume Modification Suggestions")

    if initialized and all_roles:
        mod_role = st.selectbox("Select resume by role:", all_roles, key="mod_role")

        mod_section = st.selectbox("Section to modify:",
                                   ["Skills", "Experience", "Education", "Projects", "Summary"])

        mod_action = st.selectbox("Action:", ["Add", "Update", "Delete"])

        mod_content = st.text_area("Content to add/update:",
                                   placeholder="Example: Proficient in Python and TensorFlow")

        modify_button = st.button("Generate Suggestion", disabled=not (mod_role and mod_content))

        if modify_button:
            with st.spinner("Generating modification suggestion..."):
                # Get the current content of the section
                current_section_query = query_rag(
                    qa_chain,
                    f"What is currently in the {mod_section} section of the {mod_role} resume?"
                )

                st.subheader("Modification Suggestion")

                # Get resume names
                resumes = get_resume_by_role(vectorstore, mod_role)
                resume_names = list(resumes.keys())

                # Functions to determine benefit of modification
                def get_modification_benefit(action, section, content, role):
                    """Generate explanation of modification benefit."""
                    benefits = {
                        "add": {
                            "Skills": "showcase additional capabilities relevant to the position",
                            "Experience": "highlight relevant work history that strengthens the candidate's profile",
                            "Education": "provide additional academic credentials",
                            "Projects": "demonstrate practical application of skills",
                            "Summary": "better position the candidate for the target role"
                        },
                        "update": {
                            "Skills": "better represent the candidate's skill level and focus areas",
                            "Experience": "clarify responsibilities and achievements in previous roles",
                            "Education": "provide more accurate academic information",
                            "Projects": "highlight more relevant aspects of the project work",
                            "Summary": "align the profile more closely with industry expectations"
                        },
                        "delete": {
                            "Skills": "remove outdated or irrelevant skills",
                            "Experience": "eliminate positions that aren't relevant to the target role",
                            "Education": "remove unnecessary educational details",
                            "Projects": "focus on the most relevant project experience",
                            "Summary": "eliminate unnecessary information that distracts from core qualifications"
                        }
                    }

                    # Default benefit if specific section not found
                    default_benefit = "improve the overall quality and relevance of the resume"

                    # Get the benefit for the action and section, or use the default
                    action_benefits = benefits.get(action.lower(), {})
                    return action_benefits.get(section, default_benefit)

                # Create a formatted box with the suggestion
                st.markdown(f"""
                ### MODIFICATION SUGGESTION FOR:
                **Resume:** {resume_names[0] if resume_names else mod_role}
                **Role:** {mod_role}

                **Action:** {mod_action.upper()}
                **Section:** {mod_section}

                #### CURRENT CONTENT IN SECTION:
                {current_section_query["result"]}

                #### PROPOSED CHANGE:
                {mod_content}

                #### DETAILED EXPLANATION:
                This modification would help {get_modification_benefit(mod_action, mod_section, mod_content, mod_role)}.

                > **IMPORTANT NOTE:** This is a SUGGESTION ONLY.
                > The Multi-Resume Assistant does not modify the actual PDF files.
                > To implement this change, you would need to edit the resume manually using a PDF editor.
                """)
    else:
        st.info("Please upload and process resumes first to suggest modifications.")