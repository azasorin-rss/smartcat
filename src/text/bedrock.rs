use aws_config::{meta::region::RegionProviderChain, BehaviorVersion, Region};
use aws_sdk_bedrockruntime::{config::Builder, primitives::Blob, Client};
use serde::{Deserialize, Serialize};

use crate::config::{
    api::ApiConfig,
    prompt::{Message, Prompt},
};

#[derive(Debug, Serialize, Deserialize)]
struct ClaudeResponse {
    completion: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct InferenceConfig {
    max_new_tokens: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ContentThing {
    text: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct NovaMessage {
    role: String,
    content: Vec<ContentThing>,
}

impl From<NovaMessage> for Message {
    fn from(value: NovaMessage) -> Self {
        Message::assistant(&value.content[0].text)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct NovaLiteRequest {
    #[serde(rename = "inferenceConfig")]
    inference_config: InferenceConfig,

    messages: Vec<NovaMessage>,
}

#[derive(Debug, Serialize, Deserialize)]
struct NovaLiteOutput {
    message: NovaMessage,
}

#[derive(Debug, Serialize, Deserialize)]
struct NovaLiteResponse {
    output: NovaLiteOutput,
}

impl NovaLiteRequest {
    pub fn new(content: &str) -> Self {
        Self {
            inference_config: InferenceConfig {
                max_new_tokens: 1000,
            },
            messages: vec![NovaMessage {
                role: String::from("user"),
                content: vec![ContentThing {
                    text: content.to_string(),
                }],
            }],
        }
    }
}

pub async fn bedrock(api_config: ApiConfig, prompt: &Prompt) -> anyhow::Result<Message> {
    // Load configuration
    let region_provider = RegionProviderChain::default_provider().or_else(Region::new("us-west-2")); // Replace with your preferred region
    let config = aws_config::defaults(BehaviorVersion::latest())
        .profile_name("dev")
        .region(region_provider)
        .load()
        .await;

    // Create a Bedrock client
    let bedrock_config = Builder::from(&config)
        .endpoint_url("https://bedrock-runtime.us-west-2.amazonaws.com") // Adjust endpoint if needed
        .build();
    let client = Client::from_conf(bedrock_config);

    let user_prompt = "write a very short poem";

    let request = NovaLiteRequest::new(user_prompt);

    let body = serde_json::to_string(&request).unwrap();

    // Make a request to Bedrock
    let response = client
        .invoke_model()
        .model_id("us.amazon.nova-lite-v1:0")
        .body(Blob::new(body.as_bytes()))
        .content_type("application/json")
        .send()
        .await?;

    let inner = response.body.into_inner();

    let resp = serde_json::from_slice::<NovaLiteResponse>(&inner).unwrap();

    // println!("{}", resp.output.message.content[0].text);

    Ok(resp.output.message.into())
}
