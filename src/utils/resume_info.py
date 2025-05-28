# Resume information extraction utilities
# Functions for working with resume data

def extract_resume_info(docs):
    """Extract resume information from the retrieved documents."""
    resumes = {}

    for doc in docs:
        filename = doc.metadata.get('filename', 'unknown')
        role = doc.metadata.get('role', 'Unknown Role')

        if filename not in resumes:
            resumes[filename] = {
                'role': role,
                'content': [],
                'source': doc.metadata.get('source', '')
            }

        resumes[filename]['content'].append(doc.page_content)

    return resumes

def get_resume_by_role(vectorstore, role):
    """Find a resume by role."""
    # Create a metadata filter for the role
    filter_query = {"role": {"$eq": role}}

    # Search for documents with that role
    # docs = vectorstore.similarity_search(
    #     query=f"resume for {role}",
    #     k=10,
    #     filter=filter_query
    # )

    docs = vectorstore.similarity_search(query=f"resume for {role}", k=20)
    filtered_docs = [doc for doc in docs if doc.metadata.get("role") == role]



    if not docs:
        # Try a more general search if exact match fails
        docs = vectorstore.similarity_search(
            query=f"resume for {role}",
            k=10
        )

    return extract_resume_info(docs)