import { GithubRepoLoader } from "langchain/document_loaders";
import { SupabaseVectorStore } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { createClient } from "@supabase/supabase-js";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { config } from "dotenv";
import path from "path";

config({ path: path.resolve(process.cwd(), ".env.local") });

async function run() {
  try {
    const loader = new GithubRepoLoader(
      "https://github.com/glasserviceoslo/glass.no",
      {
        branch: "netlify",
        recursive: false,
        unknown: "warn",
        accessToken: process.env.GITHUB_ACCESS_TOKEN,
      }
    );

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const docs = await loader.loadAndSplit(textSplitter);

    const client = createClient(
      process.env.SUPABASE_URL || "",
      process.env.SUPABASE_PRIVATE_KEY || ""
    );

    await SupabaseVectorStore.fromDocuments(
      docs,
      new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY }),
      {
        client,
        tableName: "documents",
      }
    );
  } catch (error: any) {
    console.log("error =>>>>> ", error.message);
    console.log("Failed to ingest your data");
  }
}

run()
  .then(() => console.log("Task completed successfully"))
  .catch((e) => console.error(e));
