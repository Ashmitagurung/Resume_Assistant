# Tab renderer functions for the main interface
# Contains functions to render each of the Streamlit tabs

import streamlit as st

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

        st.subheader("Sources:")
        for i, doc in enumerate(result["source_documents"][:3]):  # Limit to 3 sources
            with st.expander(
                    f"Source {i + 1} from {doc.metadata.get('filename', 'unknown')} ({doc.metadata.get('role', 'Unknown Role')})"):
                st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)


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
                            skills_result = query_rag(qa_chain,
                                                      f"What skills does the {selected_role} have based on {filename}?")
                            experience_result = query_rag(qa_chain,
                                                          f"What is the work experience of the {selected_role} in {filename}?")
                            education_result = query_rag(qa_chain,
                                                         f"What is the education background of the {selected_role} in {filename}?")

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
                            "Skills": "better represent the candidate's actual skill level and focus areas",
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