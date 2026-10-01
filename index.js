import { input } from "@inquirer/prompts";
import fetch from "node-fetch";
import readline from "readline";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { OllamaEmbeddings, ChatOllama } from "@langchain/ollama";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";

async function main() {
  console.log("🚀 Starting Markdown Documentation Assistant...\n");

  // 1. Prompt for markdown file URL
  let url = await input({ message: "Enter the URL to your markdown file:" });
  // Accept a pasted Markdown link and convert GitHub blob URLs to raw file URLs.
  const markdownLink = url.match(/^\[[^\]]*\]\((https?:\/\/[^)]+)\)$/);
  if (markdownLink) url = markdownLink[1];
  url = url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/");
  console.log(`\n📥 Received markdown URL: ${url}`);

  // 2. Download markdown file
  console.time("Download markdown");
  const response = await fetch(url);
  if (!response.ok) {
    console.error("❌ Failed to download file:", response.statusText);
    process.exit(1);
  }
  const markdown = await response.text();
  console.timeEnd("Download markdown");

  // 3. Chunk markdown
  console.time("Chunk markdown");
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
    separators: ["\n\n", "\n", " ", ""],
  });
  const docs = [
    new Document({ pageContent: markdown, metadata: { source: url } }),
  ];
  const chunks = await splitter.splitDocuments(docs);
  console.log(`🧩 Created ${chunks.length} chunks`);
  console.timeEnd("Chunk markdown");

  // 4. Embed and index with MemoryVectorStore
  console.time("Embedding and indexing");
  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: "http://localhost:11434",
  });

  let vectorStore;
  try {
    vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);
    console.log("✅ Vector store created successfully");
  } catch (err) {
    console.error("❌ Failed to create vector store:", err);
    process.exit(1);
  }
  console.timeEnd("Embedding and indexing");

  // 5. Set up LLM and prompt template (chatbot piece)
  console.time("LLM and prompt setup");
  const llm = new ChatOllama({
    model: "qwen2.5:0.5b",
    temperature: 0.1,
    baseUrl: "http://localhost:11434",
  });
  const promptTemplate = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are an expert documentation assistant. Use the following context to answer questions about the documentation accurately and helpfully.
Context: {context}
Guidelines:
- Provide accurate information based only on the provided context
- Include relevant code examples when available
- Mention the source document when possible
- If information is not in the context, clearly state that`,
    ],
    ["human", "{question}"],
  ]);
  console.timeEnd("LLM and prompt setup");

  // 6. CLI chat loop
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: "> ",
  });

  console.log(
    '\n💬 Ready! Ask your questions about the document. Type "exit" to quit.'
  );
  rl.prompt();

  rl.on("line", async (line) => {
    const question = line.trim();
    if (question.toLowerCase() === "exit") {
      console.log("👋 Exiting assistant. Goodbye!");
      rl.close();
      return;
    }
    try {
      console.time("Retrieval");
      const retriever = vectorStore.asRetriever({ k: 5 });
      const relevantDocs = await retriever.invoke(question);
      console.timeLog(
        "Retrieval",
        `Found ${relevantDocs.length} relevant chunks`
      );

      const context = relevantDocs.map((doc) => doc.pageContent).join("\n\n");
      console.timeEnd("Retrieval");

      console.time("Answer generation");
      const promptMessages = await promptTemplate.formatMessages({
        context: context,
        question: question,
      });
      const response = await llm.invoke(promptMessages);
      console.timeEnd("Answer generation");

      console.log(`\n📝 Answer:\n${response.content}\n`);
    } catch (err) {
      console.error("❌ Error during processing:", err.message);
    }
    rl.prompt();
  });
}

main();
