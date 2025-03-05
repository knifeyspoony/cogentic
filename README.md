# Cogentic

Cogentic is an AutoGen group chat, based on the [Magentic One](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/magentic-one.html) orchestration pattern. It differs from the original Magentic One in that it explicitly forces hypothesis creation and testing to answer a question.

## Purpose

Manages a group chat between AI agents using a scientific hypothesis-testing approach

## Key Components

- Fact management: Tracks facts and evidence
- Plan management: Maintains hypotheses and tests
- Progress tracking: Monitors conversation progress with stall detection
