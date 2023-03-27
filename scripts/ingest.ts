import { SupabaseVectorStore } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { createClient } from "@supabase/supabase-js";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { config } from "dotenv";
import path from "path";
import { GithubRepoLoader } from "./base";

config({ path: path.resolve(process.cwd(), ".env.local") });

async function run() {
  try {
    const loader = new GithubRepoLoader(
      "https://github.com/glasserviceoslo/glass.no",
      // "https://github.com/davlet61/powerofficesuiteconnector",
      {
        branch: "netlify",
        recursive: true,
        unknown: "warn",
        accessToken: process.env.GITHUB_ACCESS_TOKEN,
      }
    );

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const docs = await loader.loadAndSplit(textSplitter);

    const CHUNK_SIZE = 50;
    const chunks = Array.from(
      { length: Math.ceil(docs.length / CHUNK_SIZE) },
      (_, index) => docs.slice(index * CHUNK_SIZE, (index + 1) * CHUNK_SIZE)
    );

    const client = createClient(
      process.env.SUPABASE_URL || "",
      process.env.SUPABASE_PRIVATE_KEY || ""
    );

    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    });
    const dbOpts = { client, tableName: "documents" };

    for (const chunk of chunks) {
      await SupabaseVectorStore.fromDocuments(chunk, embeddings, dbOpts);
    }
  } catch (error: any) {
    console.log("error =>>>>> ", error.message);
    console.log("Failed to ingest your data");
  }
}

run()
  .then(() => console.log("Task completed successfully"))
  .catch((e) => console.error(e));
