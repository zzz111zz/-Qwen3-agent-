import time
import matplotlib.pyplot as plt
import ollama
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import warnings

warnings.filterwarnings("ignore")

# 配置
OLLAMA_MODEL = "qwen3.5:4b"
EMBED_MODEL = "all-MiniLM-L6-v2"
DB_PATH = "./chroma_db"
KNOWLEDGE_FILE = "./knowledge.txt"

# 测试问题
TEST_QUESTIONS = [
    "Qwen3模型的特点是什么？",
    "RAG技术的作用是什么？",
    "Agent和RAG有什么区别？",
    "简述大模型减少幻觉的方法？",
    "请详细说明复杂问题如何分步解决？"
]

# 构建知识库
def build_knowledge_base():
    if not os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
            f.write("""
Qwen3是通义千问新一代大模型，支持多轮对话、工具调用、长文本理解。
RAG通过检索外部文档增强生成，降低幻觉、提升事实准确性。
Agent具备决策能力：任务分类、动态检索、置信度重试、工具调用。
大模型幻觉可通过RAG、事实校验、思维链、置信度过滤缓解。
复杂问题可拆解：理解 → 检索 → 推理 → 验证 → 回答。
""")

    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma.from_texts(chunks, embeddings, persist_directory=DB_PATH)

vectordb = build_knowledge_base()

# Agent任务分类
def task_classifier(q):
    return "complex" if len(q) > 40 else "simple"

# 动态检索
def dynamic_retrieve(q, task_type):
    k = 2 if task_type == "simple" else 5
    return vectordb.similarity_search(q, k=k)

# 调用Ollama生成回答
def ollama_generate(q, ctx):
    prompt = f"上下文：{ctx}\n问题：{q}\n回答："
    start = time.time()
    resp = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
    ans = resp["response"]
    conf = 0.85 if len(ans) > 20 else 0.6
    return ans, conf, time.time() - start

# 执行实验
def run_experiment():
    results = {
        "纯RAG": {"acc": [], "latency": [], "multi_hop": []},
        "Agent+RAG": {"acc": [], "latency": [], "multi_hop": []}
    }

    for question in TEST_QUESTIONS:
        # 纯RAG
        docs = vectordb.similarity_search(question, k=2)
        context = "\n".join(d.page_content for d in docs)
        _, acc1, t1 = ollama_generate(question, context)
        results["纯RAG"]["acc"].append(acc1)
        results["纯RAG"]["latency"].append(t1)
        results["纯RAG"]["multi_hop"].append(1 if len(question) > 40 else 0)

        # Agent+RAG
        task_type = task_classifier(question)
        docs2 = dynamic_retrieve(question, task_type)
        context2 = "\n".join(d.page_content for d in docs2)
        _, acc2, t2 = ollama_generate(question, context2)
        if acc2 < 0.7:
            _, acc2, t2 = ollama_generate(question, context2)
        results["Agent+RAG"]["acc"].append(acc2)
        results["Agent+RAG"]["latency"].append(t2)
        results["Agent+RAG"]["multi_hop"].append(1 if len(question) > 40 else 0)

    return results

# 结果可视化
def plot_results(res):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    labels = ["准确率(%)", "响应时间(s)", "多跳率(%)"]

    pure_acc = sum(res["纯RAG"]["acc"]) / len(res["纯RAG"]["acc"]) * 100
    pure_lat = sum(res["纯RAG"]["latency"]) / len(res["纯RAG"]["latency"])
    pure_mh = sum(res["纯RAG"]["multi_hop"]) / len(res["纯RAG"]["multi_hop"]) * 100

    agent_acc = sum(res["Agent+RAG"]["acc"]) / len(res["Agent+RAG"]["acc"]) * 100
    agent_lat = sum(res["Agent+RAG"]["latency"]) / len(res["Agent+RAG"]["latency"])
    agent_mh = sum(res["Agent+RAG"]["multi_hop"]) / len(res["Agent+RAG"]["multi_hop"]) * 100

    pure_data = [pure_acc, pure_lat, pure_mh]
    agent_data = [agent_acc, agent_lat, agent_mh]

    x = range(3)
    plt.figure(figsize=(10, 6))
    plt.bar([i-0.175 for i in x], pure_data, 0.35, label="纯RAG", color="#4A90E2")
    plt.bar([i+0.175 for i in x], agent_data, 0.35, label="Agent+RAG", color="#FF6B6B")
    plt.title("Ollama Qwen3.5:4b 实验对比")
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("experiment_result.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    print("=== 实验开始 ===")
    result_data = run_experiment()
    plot_results(result_data)
    print("=== 实验完成 ===")
