from transformers import pipeline

def summarize_text(text, max_length=120, min_length=30):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]["summary_text"]

if __name__ == "__main__":
    text = """
    Artificial Intelligence has transformed multiple industries by enabling automation,
    data-driven decision making, and intelligent systems capable of learning and adapting.
    As organizations collect more data, AI-driven solutions are becoming essential for
    understanding complex patterns and solving real-world problems efficiently.
    """

    print("\nSummary:")
    print(summarize_text(text))
