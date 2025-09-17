import os
import io
import time
import gc
import json
import logging
import re
from typing import Optional, Tuple, List, Dict, Set

import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq
from peft import PeftModel

LOCAL_MODEL_PATH = "/content/drive/MyDrive/llava_lora_finetune"   # parent folder containing checkpoint-*
BASE_MODEL_NAME = "unsloth/llava-1.5-7b-hf-bnb-4bit"             # Unsloth 4-bit base
MAX_OUTPUT_TOKENS = 300

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def list_checkpoints(parent_dir: str) -> List[str]:
    """Return sorted list of checkpoint folder names (full paths)."""
    if not os.path.exists(parent_dir):
        return []
    items = [d for d in os.listdir(parent_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(parent_dir, d))]
    items = sorted(items, key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else x)
    return [os.path.join(parent_dir, d) for d in items]

def get_checkpoint_metrics(checkpoint_path: str) -> Dict:
    """Extract training metrics from trainer_state.json if available."""
    trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        try:
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)
                # Get the last logged metrics
                log_history = trainer_state.get('log_history', [])
                if log_history:
                    # Look for validation loss in the last few entries
                    for entry in reversed(log_history):
                        if 'eval_loss' in entry:
                            return {'eval_loss': entry['eval_loss'], 'step': entry.get('step', 0)}
                        elif 'train_loss' in entry:
                            return {'train_loss': entry['train_loss'], 'step': entry.get('step', 0)}
        except Exception as e:
            logger.warning(f"Could not read trainer_state.json from {checkpoint_path}: {e}")
    
    # Fallback: use checkpoint number as proxy (higher = later in training)
    try:
        step = int(os.path.basename(checkpoint_path).split("-")[-1])
        return {'step': step, 'fallback': True}
    except:
        return {'step': 0, 'fallback': True}

def select_best_checkpoint(parent_dir: str) -> Optional[str]:
    """Select the best checkpoint based on validation loss or training progression."""
    checkpoints = list_checkpoints(parent_dir)
    if not checkpoints:
        return None
    
    if len(checkpoints) == 1:
        return checkpoints[0]
    
    best_checkpoint = None
    best_metrics = None
    
    st.info("🔍 Analyzing checkpoints to find the best model...")
    
    for ckpt in checkpoints:
        metrics = get_checkpoint_metrics(ckpt)
        ckpt_name = os.path.basename(ckpt)
        
        if 'eval_loss' in metrics:
            st.text(f"  {ckpt_name}: eval_loss={metrics['eval_loss']:.4f}")
            if best_metrics is None or metrics['eval_loss'] < best_metrics.get('eval_loss', float('inf')):
                best_checkpoint = ckpt
                best_metrics = metrics
        elif 'train_loss' in metrics:
            st.text(f"  {ckpt_name}: train_loss={metrics['train_loss']:.4f}")
            if best_metrics is None or ('eval_loss' not in best_metrics and metrics['train_loss'] < best_metrics.get('train_loss', float('inf'))):
                best_checkpoint = ckpt
                best_metrics = metrics
        else:
            st.text(f"  {ckpt_name}: step={metrics['step']} (no loss data)")
            if best_metrics is None or (best_metrics.get('fallback') and metrics['step'] > best_metrics['step']):
                best_checkpoint = ckpt
                best_metrics = metrics
    
    if best_checkpoint:
        reason = "lowest eval_loss" if 'eval_loss' in best_metrics else "lowest train_loss" if 'train_loss' in best_metrics else "latest checkpoint"
        st.success(f"✅ Selected {os.path.basename(best_checkpoint)} ({reason})")
    
    return best_checkpoint

# Enhanced agent prompts with structured output instructions
AGENT_PROMPTS = {
    "therapist": {
        "system": """You are a compassionate and professional therapist specializing in relationship recovery. 
        Provide empathetic, supportive guidance focusing on emotional healing and self-care.
        
        IMPORTANT: Provide ONLY your therapeutic response. Do not repeat the user's question or prompt.
        Structure your response as advice and support, not as answers to questions.
        Keep your response focused, practical, and emotionally supportive.""",
        "title": "🤗 Emotional Support"
    },
    "closure": {
        "system": """You are a relationship closure specialist who helps people find peace and understanding after breakups.
        Focus on helping the person process their emotions, understand the situation, and find closure.
        
        IMPORTANT: Provide ONLY your closure guidance. Do not repeat the user's question or prompt.
        Offer specific steps and perspectives to help them move forward.
        Be direct but compassionate in helping them understand and accept what happened.""",
        "title": "✋ Finding Closure"
    },
    "routine": {
        "system": """You are a life coach specializing in post-breakup recovery routines and self-improvement.
        Create practical, actionable daily and weekly routines to help rebuild their life.
        
        IMPORTANT: Provide ONLY your routine recommendations. Do not repeat the user's question or prompt.
        Structure your response as concrete steps and schedules they can follow.
        Focus on rebuilding habits, social connections, and personal growth.""",
        "title": "📅 Your Recovery Plan"
    },
    "honesty": {
        "system": """You are a brutally honest but caring friend who provides direct, unfiltered perspective on relationships.
        Give tough love advice that helps people see reality clearly and take accountability.
        
        IMPORTANT: Provide ONLY your honest assessment. Do not repeat the user's question or prompt.
        Be direct and frank while still being supportive of their growth.
        Help them see patterns, take responsibility, and make better choices moving forward.""",
        "title": "💪 Honest Perspective"
    }
}

# Agent selection criteria based on query analysis
AGENT_SELECTION_CRITERIA = {
    "therapist": {
        "keywords": ["sad", "depressed", "hurt", "crying", "emotional", "feelings", "overwhelmed", 
                    "anxiety", "stress", "support", "comfort", "healing", "pain", "heartbreak"],
        "patterns": ["i feel", "i'm feeling", "emotionally", "can't stop", "so sad", "devastated"],
        "sentiment": "negative_emotional",
        "description": "Emotional support and therapeutic guidance"
    },
    "closure": {
        "keywords": ["why", "understand", "closure", "confused", "questions", "answers", "explain",
                    "don't get", "makes no sense", "suddenly", "out of nowhere", "blindsided"],
        "patterns": ["why did", "i don't understand", "what happened", "need closure", "so confused"],
        "sentiment": "confused_seeking",
        "description": "Help understanding and finding closure"
    },
    "routine": {
        "keywords": ["routine", "schedule", "daily", "activities", "productive", "goals", "plan",
                    "structure", "habits", "lifestyle", "rebuild", "moving forward", "future"],
        "patterns": ["what should i do", "how do i move", "need structure", "daily routine"],
        "sentiment": "action_oriented",
        "description": "Practical recovery plans and routines"
    },
    "honesty": {
        "keywords": ["fault", "blame", "mistake", "wrong", "honest", "truth", "reality", "advice",
                    "tell me straight", "brutal", "real talk", "accountability", "my part"],
        "patterns": ["was it my fault", "what did i do wrong", "be honest", "tell me the truth"],
        "sentiment": "accountability_seeking",
        "description": "Direct, honest perspective and tough love"
    }
}

def analyze_query_rule_based(user_input: str) -> Dict[str, float]:
    """Rule-based analysis using keywords and patterns."""
    user_lower = user_input.lower()
    agent_scores = {agent: 0.0 for agent in AGENT_SELECTION_CRITERIA}
    
    # Keyword matching
    for agent, criteria in AGENT_SELECTION_CRITERIA.items():
        keyword_matches = sum(1 for keyword in criteria["keywords"] if keyword in user_lower)
        pattern_matches = sum(1 for pattern in criteria["patterns"] if pattern in user_lower)
        
        # Calculate base score
        base_score = (keyword_matches * 0.6) + (pattern_matches * 1.0)
        
        # Apply sentiment bonuses
        sentiment_bonus = 0.0
        if criteria["sentiment"] == "negative_emotional":
            # Look for emotional distress indicators
            emotional_indicators = ["can't", "won't", "never", "always", "everything", "nothing"]
            sentiment_bonus = sum(0.3 for indicator in emotional_indicators if indicator in user_lower)
            
        elif criteria["sentiment"] == "confused_seeking":
            # Look for confusion and question words
            question_words = ["why", "how", "what", "when", "where"]
            sentiment_bonus = sum(0.4 for word in question_words if word in user_lower)
            
        elif criteria["sentiment"] == "action_oriented":
            # Look for forward-thinking language
            future_words = ["will", "going to", "want to", "need to", "should", "plan"]
            sentiment_bonus = sum(0.3 for word in future_words if word in user_lower)
            
        elif criteria["sentiment"] == "accountability_seeking":
            # Look for self-reflection language
            self_words = ["i did", "my fault", "i was", "i should", "i could"]
            sentiment_bonus = sum(0.5 for word in self_words if word in user_lower)
        
        agent_scores[agent] = base_score + sentiment_bonus
    
    return agent_scores

def analyze_query_llm_based(model, tokenizer, user_input: str) -> Dict[str, float]:
    """LLM-based analysis for more nuanced understanding."""
    analysis_prompt = f"""Analyze this breakup/relationship situation and determine which types of support would be most helpful.

User situation: "{user_input}"

For each support type, rate the relevance from 0.0 to 5.0:

1. THERAPIST (Emotional support, healing, comfort): Focus on emotional pain, sadness, trauma, need for validation
2. CLOSURE (Understanding, answers, closure): Focus on confusion, seeking explanations, need to understand what happened  
3. ROUTINE (Life planning, structure, moving forward): Focus on practical next steps, rebuilding life, future planning
4. HONESTY (Direct feedback, accountability, tough love): Focus on self-reflection, wanting honest perspective, taking responsibility

Respond in this exact format:
THERAPIST: X.X
CLOSURE: X.X  
ROUTINE: X.X
HONESTY: X.X

Only provide the ratings, no explanations."""

    try:
        inputs = tokenizer(analysis_prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,  # Use deterministic generation
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the rating part from response
        if "THERAPIST:" in response:
            response = response.split("THERAPIST:")[-1]
        
        # Parse the LLM response
        agent_scores = {"therapist": 0.0, "closure": 0.0, "routine": 0.0, "honesty": 0.0}
        
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    agent_name = parts[0].strip().lower()
                    try:
                        score = float(parts[1].strip())
                        if agent_name == "therapist" and "therapist" in agent_scores:
                            agent_scores["therapist"] = min(max(score, 0.0), 5.0)
                        elif agent_name == "closure" and "closure" in agent_scores:
                            agent_scores["closure"] = min(max(score, 0.0), 5.0)
                        elif agent_name == "routine" and "routine" in agent_scores:
                            agent_scores["routine"] = min(max(score, 0.0), 5.0)
                        elif agent_name == "honesty" and "honesty" in agent_scores:
                            agent_scores["honesty"] = min(max(score, 0.0), 5.0)
                    except ValueError:
                        continue
        
        return agent_scores
        
    except Exception as e:
        logger.warning(f"LLM analysis failed: {e}")
        # Fallback to neutral scores
        return {"therapist": 2.0, "closure": 1.0, "routine": 1.0, "honesty": 1.0}

def analyze_query_hybrid(model, tokenizer, user_input: str, rule_weight: float = 0.3, llm_weight: float = 0.7) -> Dict[str, float]:
    """Hybrid analysis combining rule-based and LLM-based approaches."""
    # Get rule-based scores
    rule_scores = analyze_query_rule_based(user_input)
    
    # Get LLM-based scores  
    llm_scores = analyze_query_llm_based(model, tokenizer, user_input)
    
    # Normalize rule-based scores to 0-5 scale to match LLM scores
    max_rule_score = max(rule_scores.values()) if max(rule_scores.values()) > 0 else 1
    normalized_rule_scores = {agent: (score / max_rule_score) * 5.0 for agent, score in rule_scores.items()}
    
    # Combine scores with weighted average
    hybrid_scores = {}
    for agent in rule_scores:
        rule_component = normalized_rule_scores[agent] * rule_weight
        llm_component = llm_scores.get(agent, 0.0) * llm_weight
        hybrid_scores[agent] = rule_component + llm_component
    
    return hybrid_scores, rule_scores, llm_scores

def select_relevant_agents(model, tokenizer, user_input: str, min_threshold: float = 2.0, rule_weight: float = 0.3, llm_weight: float = 0.7) -> Tuple[List[str], Dict]:
    """Select the most relevant agents using hybrid analysis."""
    # Get hybrid scores and component scores
    hybrid_scores, rule_scores, llm_scores = analyze_query_hybrid(model, tokenizer, user_input, rule_weight, llm_weight)
    
    # Sort agents by hybrid relevance score
    sorted_agents = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Always include at least the top agent if it has any relevance
    selected_agents = []
    
    if sorted_agents[0][1] > 0.5:  # Minimum relevance threshold
        selected_agents.append(sorted_agents[0][0])
    
    # Add other relevant agents based on threshold
    for agent, score in sorted_agents[1:]:
        if score >= min_threshold:
            selected_agents.append(agent)
    
    # Fallback: if no clear matches, provide therapeutic support
    if not selected_agents:
        selected_agents = ["therapist"]
    
    # Limit to max 3 agents to avoid overwhelming response
    selected_agents = selected_agents[:3]
    
    # Return analysis details for transparency
    analysis_details = {
        "hybrid_scores": hybrid_scores,
        "rule_scores": rule_scores, 
        "llm_scores": llm_scores,
        "weights": {"rule_weight": rule_weight, "llm_weight": llm_weight}
    }
    
    return selected_agents, analysis_details

def explain_agent_selection_hybrid(selected_agents: List[str], analysis_details: Dict) -> str:
    """Generate detailed explanation for hybrid agent selection."""
    explanations = []
    
    agent_names = {
        "therapist": "🤗 Emotional Support",
        "closure": "✋ Closure Guidance", 
        "routine": "📅 Recovery Planning",
        "honesty": "💪 Honest Perspective"
    }
    
    hybrid_scores = analysis_details["hybrid_scores"]
    rule_scores = analysis_details["rule_scores"]
    llm_scores = analysis_details["llm_scores"]
    weights = analysis_details["weights"]
    
    explanations.append("**Analysis Breakdown:**")
    explanations.append(f"*Combining Rule-based ({weights['rule_weight']:.1%}) + LLM Analysis ({weights['llm_weight']:.1%})*")
    explanations.append("")
    
    for agent in selected_agents:
        agent_name = agent_names[agent]
        hybrid_score = hybrid_scores[agent]
        rule_score = rule_scores.get(agent, 0.0)
        llm_score = llm_scores.get(agent, 0.0)
        description = AGENT_SELECTION_CRITERIA[agent]["description"]
        
        explanations.append(f"**{agent_name}** - Final Score: {hybrid_score:.1f}/5.0")
        explanations.append(f"  • Rule-based: {rule_score:.1f} | LLM Analysis: {llm_score:.1f}")
        explanations.append(f"  • {description}")
        explanations.append("")
    
    return "\n".join(explanations)

@st.cache_resource
def load_model_and_processors(checkpoint_path: str) -> Tuple[Optional[PeftModel], Optional[AutoProcessor], Optional[AutoTokenizer]]:
    """Load base Unsloth model (4-bit) and PEFT checkpoint (LoRA) from checkpoint_path."""
    try:
        clear_gpu_memory()
        if not os.path.exists(checkpoint_path):
            st.error(f"Checkpoint not found: {checkpoint_path}")
            return None, None, None

        st.info(f"Loading tokenizer & processor from {BASE_MODEL_NAME} ...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
        processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)

        st.info("Loading base model (Unsloth 4-bit) ...")
        base_model = AutoModelForVision2Seq.from_pretrained(
            BASE_MODEL_NAME,
            load_in_4bit=True,
            device_map="auto"
        )

        st.info(f"Loading LoRA from checkpoint: {os.path.basename(checkpoint_path)} ...")
        peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
        peft_model = peft_model.merge_and_unload()
        peft_model.eval()

        st.success("✅ Model + adapters loaded")
        return peft_model, processor, tokenizer

    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        st.error(f"Error loading model: {e}")
        return None, None, None

def clean_response(response: str, original_prompt: str = "") -> str:
    """Clean the model response by removing repeated questions and formatting properly."""
    # Remove the original prompt from the response if it appears
    if original_prompt and original_prompt.strip() in response:
        response = response.replace(original_prompt.strip(), "").strip()
    
    # Remove common question patterns that the model might repeat
    lines = response.split('\n')
    cleaned_lines = []
    
    skip_patterns = [
        "what would you like to know",
        "how can I help",
        "what's your question",
        "tell me more about",
        "can you share",
        "what happened",
        "how are you feeling"
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip lines that are just repeated questions
        line_lower = line.lower()
        should_skip = False
        for pattern in skip_patterns:
            if pattern in line_lower and line_lower.endswith('?'):
                should_skip = True
                break
        
        if not should_skip:
            cleaned_lines.append(line)
    
    # Join lines and ensure proper formatting
    cleaned_response = '\n\n'.join(cleaned_lines)
    
    # Remove any remaining duplicate sentences
    sentences = cleaned_response.split('. ')
    unique_sentences = []
    seen = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence.lower() not in seen:
            seen.add(sentence.lower())
            unique_sentences.append(sentence)
    
    final_response = '. '.join(unique_sentences)
    if final_response and not final_response.endswith('.'):
        final_response += '.'
    
    return final_response

def generate_structured_response(model, processor, tokenizer, agent_key: str, user_input: str, image: Optional[Image.Image] = None) -> str:
    """Generate a structured response using agent-specific prompts."""
    agent_config = AGENT_PROMPTS[agent_key]
    
    # Create structured prompt
    structured_prompt = f"{agent_config['system']}\n\nUser situation: {user_input}\n\nYour response:"
    
    try:
        if image is not None:
            # Text + Image mode
            inputs = processor(text=structured_prompt, images=image, return_tensors="pt").to(model.device)
        else:
            # Text only mode
            inputs = tokenizer(structured_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_OUTPUT_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                repetition_penalty=1.2,
                early_stopping=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the response
        if "Your response:" in response:
            response = response.split("Your response:")[-1].strip()
        
        # Clean the response
        cleaned_response = clean_response(response, user_input)
        
        return cleaned_response if cleaned_response else "I understand your situation and I'm here to help you through this difficult time."
        
    except Exception as e:
        logger.error(f"Error generating response for {agent_key}: {e}")
        return f"I apologize, but I encountered an error while generating advice. Please try again."

# Streamlit UI
st.set_page_config(page_title="💔 Breakup Recovery Squad (Auto-Select Best Model)", layout="wide", page_icon="💔")
st.title("💔 Breakup Recovery Squad")
st.markdown("*Powered by automatically selected best checkpoint*")

# Show GPU info
if torch.cuda.is_available():
    try:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        st.info(f"🎯 Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    except Exception:
        pass
else:
    st.warning("⚠️ No GPU detected — inference will be very slow or may fail.")

# Auto-select best checkpoint
with st.spinner("🔍 Finding and loading the best model checkpoint..."):
    best_checkpoint = select_best_checkpoint(LOCAL_MODEL_PATH)
    
    if not best_checkpoint:
        st.error("No checkpoint folders found. Make sure you have trained the model and the path is correct.")
        st.stop()
    
    # Load the best model
    model, processor, tokenizer = load_model_and_processors(best_checkpoint)
    if model is None:
        st.error("Failed to load the best model. Check the logs above.")
        st.stop()

st.success(f"🎯 Using best checkpoint: {os.path.basename(best_checkpoint)}")

# Main UI
col1, col2 = st.columns(2)
with col1:
    user_input = st.text_area("Share your feelings or situation:", height=150, 
                             placeholder="Tell me what happened or how you're feeling...")
with col2:
    uploaded_files = st.file_uploader("Upload chat screenshots (optional)", 
                                     type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Show AI agent selection preview
if user_input.strip():
    with st.expander("🧠 AI Agent Selection Preview", expanded=False):
        with st.spinner("Analyzing your input..."):
            preview_agents, preview_details = select_relevant_agents(model, tokenizer, user_input, min_threshold=2.0)
            preview_explanation = explain_agent_selection_hybrid(preview_agents, preview_details)
        
        st.markdown("**Based on your input, I would recommend:**")
        st.markdown(preview_explanation)
        st.markdown("*This is a preview - click 'Get Recovery Plan' below to get full responses*")

# Advanced options
with st.expander("⚙️ Advanced Options", expanded=False):
    col_adv1, col_adv2 = st.columns(2)
    
    with col_adv1:
        sensitivity = st.slider(
            "Agent Selection Sensitivity", 
            min_value=1.0, max_value=4.0, value=2.0, step=0.2,
            help="Lower = more agents selected, Higher = fewer, more relevant agents"
        )
        
        rule_weight = st.slider(
            "Rule-based Weight",
            min_value=0.1, max_value=0.9, value=0.3, step=0.1,
            help="Weight for keyword/pattern matching"
        )
    
    with col_adv2:
        llm_weight = st.slider(
            "LLM Analysis Weight", 
            min_value=0.1, max_value=0.9, value=0.7, step=0.1,
            help="Weight for AI-based understanding"
        )
        
        manual_override = st.checkbox(
            "Manual Agent Override", 
            help="Check this to manually select agents instead of AI selection"
        )
    
    # Ensure weights sum to 1.0
    if abs(rule_weight + llm_weight - 1.0) > 0.01:
        total_weight = rule_weight + llm_weight
        rule_weight = rule_weight / total_weight
        llm_weight = llm_weight / total_weight
        st.info(f"Weights normalized: Rule-based: {rule_weight:.1%}, LLM: {llm_weight:.1%}")
    
    if manual_override:
        selected_agents_manual = st.multiselect(
            "Manually select recovery agents:",
            ["🤗 Therapist", "✋ Closure Specialist", "📅 Routine Planner", "💪 Honest Feedback"],
            default=["🤗 Therapist"],
            help="Manual selection overrides AI recommendations"
        )

if st.button("Get Recovery Plan 💙", type="primary"):
    if not user_input.strip():
        st.warning("Please share your situation or feelings to get personalized advice.")
    else:
        # Process uploaded image if available
        image_input = None
        if uploaded_files:
            try:
                image_input = Image.open(io.BytesIO(uploaded_files[0].read())).convert("RGB")
                st.info(f"📸 Analyzing uploaded image: {uploaded_files[0].name}")
            except Exception as e:
                st.error(f"Could not process image: {e}")
                image_input = None

        # Determine which agents to use
        if manual_override and 'selected_agents_manual' in locals():
            # Use manual selection
            agent_key_map = {
                "🤗 Therapist": "therapist",
                "✋ Closure Specialist": "closure", 
                "📅 Routine Planner": "routine",
                "💪 Honest Feedback": "honesty"
            }
            selected_agents_keys = [agent_key_map[agent] for agent in selected_agents_manual]
            selection_method = "Manual Selection"
            analysis_details = None
        else:
            # Use hybrid AI selection
            with st.spinner("🧠 Analyzing your situation with hybrid AI..."):
                selected_agents_keys, analysis_details = select_relevant_agents(
                    model, tokenizer, user_input, 
                    min_threshold=sensitivity, 
                    rule_weight=rule_weight, 
                    llm_weight=llm_weight
                )
            selection_method = "Hybrid AI Selection"

        # Show selection explanation
        st.header("Your Personalized Recovery Plan")
        
        with st.expander(f"🎯 {selection_method} - Why These Agents?", expanded=True):
            if analysis_details:
                explanation = explain_agent_selection_hybrid(selected_agents_keys, analysis_details)
                st.markdown(explanation)
                st.markdown(f"*Sensitivity threshold: {sensitivity:.1f}/5.0*")
            else:
                st.markdown("**Manually Selected Agents:**")
                agent_display_map = {
                    "therapist": "🤗 Emotional Support",
                    "closure": "✋ Closure Guidance", 
                    "routine": "📅 Recovery Planning",
                    "honesty": "💪 Honest Perspective"
                }
                for agent_key in selected_agents_keys:
                    st.markdown(f"• **{agent_display_map[agent_key]}**")

        st.markdown("---")

        # Generate responses from selected agents
        for i, agent_key in enumerate(selected_agents_keys):
            agent_title = AGENT_PROMPTS[agent_key]["title"]
            
            with st.spinner(f"Generating {agent_title} response..."):
                try:
                    response = generate_structured_response(
                        model, processor, tokenizer, agent_key, user_input, image_input
                    )
                    
                    st.subheader(agent_title)
                    st.markdown(response)
                    
                    # Add spacing between agents
                    if i < len(selected_agents_keys) - 1:
                        st.markdown("---")
                    
                    time.sleep(0.5)  # Small delay for better UX
                    
                except Exception as e:
                    st.error(f"Error generating {agent_title} response: {e}")

# Utility buttons
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("🧹 Clear GPU Memory"):
        clear_gpu_memory()
        st.success("GPU memory cleared!")

with col_btn2:
    if st.button("🔄 Reload Best Model"):
        st.cache_resource.clear()
        st.rerun()

# Debug info
with st.expander("🔧 Debug Information"):
    st.write(f"**Model Path:** `{LOCAL_MODEL_PATH}`")
    st.write(f"**Selected Checkpoint:** `{os.path.basename(best_checkpoint) if best_checkpoint else 'None'}`")
    
    all_checkpoints = list_checkpoints(LOCAL_MODEL_PATH)
    if all_checkpoints:
        st.write("**Available Checkpoints:**")
        for ckpt in all_checkpoints:
            metrics = get_checkpoint_metrics(ckpt)
            status = "📍 **SELECTED**" if ckpt == best_checkpoint else ""
            if 'eval_loss' in metrics:
                st.write(f"  - {os.path.basename(ckpt)}: eval_loss={metrics['eval_loss']:.4f} {status}")
            elif 'train_loss' in metrics:
                st.write(f"  - {os.path.basename(ckpt)}: train_loss={metrics['train_loss']:.4f} {status}")
            else:
                st.write(f"  - {os.path.basename(ckpt)}: step={metrics['step']} {status}")

st.markdown("---")
st.markdown("<div style='text-align:center'>Made with ❤️ by the Breakup Recovery Squad</div>", unsafe_allow_html=True)