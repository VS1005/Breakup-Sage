# Breakup-Sage - Agent Architecture Document

---

## Architecture Components

### 1. Frontend Layer
- **User Interface**: Streamlit-based web application
- **Input Processing**: Text input and optional image upload (chat screenshots)
- **Configuration Panel**: Sensitivity controls and manual overrides
- **Response Display**: Formatted agent responses with selection explanations

### 2. Agent Selection Engine
- **Rule-Based Analyzer**: Fast keyword and pattern matching
- **LLM-Based Analyzer**: Contextual understanding using the base model
- **Hybrid Combiner**: Weighted combination of both analysis methods (30% rule + 70% LLM)
- **Selection Logic**: Automatically chooses 1-3 most relevant agents

### 3. Specialized Recovery Agents
- **Therapist**: Emotional support and healing guidance
- **Closure Specialist**: Understanding and acceptance assistance
- **Routine Planner**: Practical life rebuilding strategies
- **Honest Feedback**: Direct perspective and accountability

### 4. Model Infrastructure
- **Base Model**: LLaVA-1.5-7B with 4-bit quantization
- **Fine-tuning**: PEFT with LoRA adapters
- **Checkpoint Management**: Automatic best checkpoint selection
- **Memory Management**: GPU optimization and cleanup

---

## Interaction Flow

### User POV :
1. **Input Collection**: User shares situation and optionally uploads images
2. **Parallel Analysis**: 
   - Rule-based system scans for keywords and patterns
   - LLM analyzes emotional context and needs
3. **Score Combination**: Weighted hybrid scoring for each agent
4. **Agent Selection**: Top scoring agents above threshold (max 3)
5. **Response Generation**: Selected agents generate specialized responses
6. **Display Results**: Formatted responses with selection explanation

### Selection Process
- **Primary Agent**: Always include highest scoring agent if above minimum threshold
- **Additional Agents**: Include others only if they exceed additional threshold
- **Fallback**: Default to Therapist agent if no clear matches
- **Transparency**: Show analysis breakdown and selection reasoning

---

## Models Used

### Core Model: LLaVA-1.5-7B
- **Architecture**: Vision-language model combining CLIP vision encoder with Vicuna-7B language model
- **Quantization**: 4-bit quantization for memory efficiency
- **Capabilities**: Handles both text and images (chat screenshots)

### Fine-tuning Strategy: PEFT + LoRA
- **Method**: Low-Rank Adaptation on specific model layers
- **Parameters**: Rank=16, Alpha=32, targeting attention and feed-forward layers
- **Efficiency**: Only trains 0.14% of total parameters while maintaining quality

### Agent Selection Models
- **Rule-Based Component**: Predefined keyword dictionaries and pattern matching
- **LLM Component**: Specialized prompts sent to the fine-tuned model for 0-5 scoring
- **Hybrid Scoring**: Weighted combination normalized to common scale

---

## Design Rationale

### Multi-Agent Architecture
- **Specialization**: Each agent focuses on specific recovery aspects for targeted support
- **Modularity**: Easy to add new agents or modify existing ones independently
- **User Experience**: Provides comprehensive support without overwhelming users

### Hybrid Selection System
- **Accuracy**: LLM provides nuanced understanding while rules ensure consistent baseline
- **Speed**: Rule-based component offers fast initial analysis
- **Transparency**: Users understand why specific agents were chosen

### LLaVA-1.5-7B Choice
- **Multi-modal**: Essential for analyzing chat screenshots and relationship images
- **Size Balance**: 7B parameters provide good quality without excessive resource requirements

### Automatic Agent Selection
- **Reduced Cognitive Load**: Distressed users don't need to make complex decisions
- **Multiple Perspectives**: Can activate multiple agents when situation requires comprehensive support
- **User Control**: Manual override available for users who prefer direct control

---

## System Benefits

### For Users
- **Personalized Support**: Automatic matching to most relevant recovery approaches
- **Reduced Decision Fatigue**: No need to choose between confusing options
- **Transparent Process**: Clear explanations for AI decisions
