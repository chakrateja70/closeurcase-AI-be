import json
text = """
This is a sample case text that is long enough to be summarized.
"""
document_urls = [
    "https://example.com/case1.pdf",
]
print(json.dumps({"case_text": text, "document_urls": document_urls}))