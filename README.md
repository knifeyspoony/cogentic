# Cogentic

Cogentic is an AutoGen group chat, based on the [Magentic One](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/magentic-one.html) orchestration pattern. It differs from the original Magentic One in that it explicitly forces hypothesis creation and testing to answer a question.

## Purpose

Manages a group chat between AI agents using a scientific hypothesis-testing approach.

## Key Components

- Uses pydantic models for data validation and serialization
- Fact management: Tracks facts and evidence during the conversation
- Plan management: Maintains hypotheses and tests, and re-plans whenever a stall occurs or a test is completed

## Installation

```bash
$> pip install cogentic
```

## Usage

To create a model client for cogentic, you'll need to be familiar with autogen.

See [AutoGen AgentChat Quickstart](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/quickstart.html) for details on how to set up your environment and create a model client.

You can then follow the [sample](./examples/sample.py) to get started with a CogenticGroupChat.

## Development Installation

```bash
$> git clone https://github.com/knifeyspoony/cogentic.git
$> cd cogentic
$> uv sync
```


