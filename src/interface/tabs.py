# Tab renderer functions for the main interface
# Contains functions to render each of the Streamlit tabs

import streamlit as st
import re
from langchain.schema import Document


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


def query_specific_resume(qa_chain, vectorstore, query, filename):
    """Query specific resume using metadata filtering"""
    try:
        # Create a custom retriever that filters by filename
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "filter": {"filename": filename},
                "k": 5
            }
        )

        # Get relevant documents for this specific file
        relevant_docs = retriever.get_relevant_documents(query)

        if not relevant_docs:
            return {"result": f"No relevant information found in {filename} for this query.", "source_documents": []}

        # Create a custom prompt context from filtered documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Create a more specific query
        specific_query = f"Based only on the resume {filename}, {query}"

        # Run the query with filtered context
        result = qa_chain({"query": specific_query, "context": context})
        result["source_documents"] = relevant_docs

        return result
    except Exception as e:
        # Fallback to original method if filtering fails
        st.warning(f"Filtering failed, using fallback method: {str(e)}")
        return qa_chain({"query": f"From {filename} only: {query}"})


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
                    st.success(f"Found {len(resumes)} resume(s) for role: {selected_role}")

                    for filename, info in resumes.items():
                        # Verify the role matches exactly
                        if info['role'].lower() != selected_role.lower():
                            continue  # Skip if role doesn't match exactly

                        st.subheader(f"📄 Resume: {filename}")
                        st.write(f"**Role:** {info['role']}")

                        # Combine the content chunks and show a preview
                        full_content = " ".join(info['content'])
                        with st.expander("Resume Content Preview"):
                            st.write(full_content[:1000] + "..." if len(full_content) > 1000 else full_content)

                        # Run specific queries for this exact resume using metadata filtering
                        with st.spinner(f"Analyzing {filename} details..."):

                            # Create more specific queries that target only this file
                            skills_query = f"List all the technical skills, programming languages, and tools mentioned in this resume."
                            experience_query = f"Describe the work experience and professional background mentioned in this resume."
                            education_query = f"List the educational qualifications and degrees mentioned in this resume."

                            # Use the new query_specific_resume function for better filtering
                            try:
                                skills_result = query_specific_resume(qa_chain, vectorstore, skills_query, filename)
                                experience_result = query_specific_resume(qa_chain, vectorstore, experience_query,
                                                                          filename)
                                education_result = query_specific_resume(qa_chain, vectorstore, education_query,
                                                                         filename)
                            except:
                                # Fallback to original method with more specific queries
                                skills_result = query_rag(qa_chain,
                                                          f"From the resume file '{filename}' ONLY, what are the technical skills? Do not include information from other resumes.")
                                experience_result = query_rag(qa_chain,
                                                              f"From the resume file '{filename}' ONLY, what is the work experience? Do not include information from other resumes.")
                                education_result = query_rag(qa_chain,
                                                             f"From the resume file '{filename}' ONLY, what is the education background? Do not include information from other resumes.")

                            # Display results in columns
                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("##### 🛠️ Skills")
                                with st.container():
                                    st.write(skills_result["result"])

                                st.markdown("##### 🎓 Education")
                                with st.container():
                                    st.write(education_result["result"])

                            with col2:
                                st.markdown("##### 💼 Experience")
                                with st.container():
                                    st.write(experience_result["result"])

                        # Add a separator between resumes
                        st.markdown("---")

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