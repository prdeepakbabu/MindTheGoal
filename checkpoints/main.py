#!/usr/bin/env python3
"""MindTheGoal CLI - Goal Success Rate evaluation for multi-turn conversations."""

import asyncio
import argparse
import logging
import sys
import json
from typing import Optional, List
from pathlib import Path
from datetime import datetime

from config import get_settings


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_checkpoint(checkpoint_file: Path) -> dict:
    """Load checkpoint from file if it exists."""
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    return {"completed_indices": [], "evaluated_sessions": []}


def save_checkpoint(checkpoint_file: Path, checkpoint_data: dict):
    """Save checkpoint to file."""
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f)


async def run_evaluation(
    dataset: str,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
    output_file: Optional[str] = None,
    data_dir: Optional[str] = None,
    resume: bool = False
):
    """Run evaluation on a dataset with checkpointing support."""
    from datasets.registry import get_loader
    from agents.judge_agent import JudgeAgent
    from core.goal_segmentation import GoalSegmenter
    from core.gsr_calculator import GSRCalculator
    from core.models import Session
    
    settings = get_settings()
    
    print(f"\n{'='*65}")
    print("           MindTheGoal Evaluation Framework")
    print(f"{'='*65}")
    print(f"\nDataset: {dataset}")
    print(f"Sample size: {sample_size or 'Full dataset'}")
    print(f"Random seed: {random_seed}")
    print(f"\n{'─'*65}")
    print("Loading dataset...")
    
    # Load dataset
    loader = get_loader(
        name=dataset,
        data_dir=data_dir or settings.get_datasets_path(dataset)
    )
    
    if sample_size:
        sessions = await loader.load_sample(
            sample_size=sample_size,
            random_seed=random_seed
        )
    else:
        sessions = await loader.load()
    
    print(f"✓ Loaded {len(sessions)} sessions")
    
    # Setup checkpointing
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_file = checkpoint_dir / f"{dataset}_{sample_size or 'full'}_{random_seed}.json"
    
    checkpoint = {"completed_indices": [], "evaluated_sessions": []}
    if resume and checkpoint_file.exists():
        checkpoint = load_checkpoint(checkpoint_file)
        print(f"✓ Resuming from checkpoint ({len(checkpoint['completed_indices'])} sessions already done)")
    
    # Initialize components
    log_dir = Path("logs/evaluations") / f"{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    judge = JudgeAgent(log_dir=str(log_dir))
    segmenter = GoalSegmenter()
    calculator = GSRCalculator()
    
    print(f"✓ Detailed logs will be saved to: {log_dir}")
    print(f"\n{'─'*65}")
    print("Evaluating sessions...")
    
    evaluated_sessions: List[Session] = []
    total = len(sessions)
    total_goals = 0
    
    # Restore previously evaluated sessions
    if checkpoint["evaluated_sessions"]:
        for session_data in checkpoint["evaluated_sessions"]:
            restored_session = Session.from_dict(session_data)
            evaluated_sessions.append(restored_session)
            total_goals += len(restored_session.goals)
    
    for i, session in enumerate(sessions):
        # Skip if already evaluated
        if i in checkpoint["completed_indices"]:
            continue
        
        # Progress indicator
        progress = (i + 1) / total * 100
        sys.stdout.write(f"\r  Progress: {i+1}/{total} ({progress:.1f}%) | Goals detected: {total_goals}")
        sys.stdout.flush()
        
        try:
            # Evaluate with session index for logging
            evaluations = await judge.evaluate_session(session, session_index=i)
            session = await judge.apply_evaluations(session, evaluations)
            session = segmenter.segment_session(session)
            
            evaluated_sessions.append(session)
            total_goals += len(session.goals)
            
            # Update checkpoint
            checkpoint["completed_indices"].append(i)
            checkpoint["evaluated_sessions"].append(session.to_dict())
            save_checkpoint(checkpoint_file, checkpoint)
            
        except Exception as e:
            logging.error(f"Error evaluating session {i}: {e}")
            # Continue with other sessions
            continue
    
    print(f"\n✓ Evaluated all sessions")
    print(f"✓ Total goals detected: {total_goals}")
    
    # Calculate GSR
    result = calculator.calculate_dataset_gsr(evaluated_sessions)
    result.dataset_name = dataset
    report = calculator.generate_report(result)
    
    # Print report
    print(f"\n{calculator.print_report(report)}")
    
    # Save detailed results
    if output_file:
        # Enhanced output with session details
        detailed_result = result.to_dict()
        detailed_result["session_details"] = []
        
        for session in evaluated_sessions:
            session_detail = {
                "session_id": session.session_id,
                "num_turns": len(session.turns),
                "num_goals": len(session.goals),
                "goals": []
            }
            for goal in session.goals:
                goal_detail = {
                    "goal_number": goal.goal_number,
                    "num_turns": len(goal.turns),
                    "quality": goal.quality.value if goal.quality else "unknown",
                    "rcof": goal.rcof
                }
                session_detail["goals"].append(goal_detail)
            detailed_result["session_details"].append(session_detail)
        
        with open(output_file, "w") as f:
            json.dump(detailed_result, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")
        print(f"✓ Detailed logs saved to {log_dir}")
    
    # Clean up checkpoint on success
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    return result


def run_server(host: str, port: int, reload: bool = False):
    """Run the API server."""
    import uvicorn
    
    print(f"\n{'='*65}")
    print("           MindTheGoal API Server")
    print(f"{'='*65}")
    print(f"\nStarting server at http://{host}:{port}")
    print(f"API docs: http://{host}:{port}/docs")
    print(f"\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload
    )


async def run_chat(domain: Optional[str] = None, evaluate: bool = True):
    """Run interactive chat with evaluation."""
    from agents.chat_agent import ChatAgent, TaskOrientedChatAgent
    from agents.judge_agent import JudgeAgent
    
    print(f"\n{'='*65}")
    print("           MindTheGoal Interactive Chat")
    print(f"{'='*65}")
    
    if domain:
        print(f"\nDomain: {domain}")
        agent = TaskOrientedChatAgent(domain=domain)
    else:
        print("\nDomain: General")
        agent = ChatAgent()
    
    print(f"Evaluation: {'Enabled' if evaluate else 'Disabled'}")
    print("\nType 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to start a new conversation")
    print(f"\n{'─'*65}\n")
    
    judge = JudgeAgent() if evaluate else None
    history = []
    turn_number = 0
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit"]:
                print("\nGoodbye!")
                break
            
            if user_input.lower() == "clear":
                history = []
                turn_number = 0
                agent.clear_history()
                print("\n[Conversation cleared]\n")
                continue
            
            turn_number += 1
            
            # Get response
            response = await agent.respond(user_input, history)
            print(f"\nAssistant: {response}\n")
            
            # Evaluate if enabled
            if judge:
                evaluation = await judge.evaluate_single_response(
                    user_message=user_input,
                    agent_response=response,
                    conversation_history=history
                )
                
                quality = evaluation.get("quality", "unknown")
                emoji = "✓" if quality == "success" else "✗"
                rcof = evaluation.get("rcof", "")
                rcof_str = f" [{rcof}]" if rcof else ""
                
                print(f"  [{emoji} {quality.upper()}{rcof_str}]")
                if evaluation.get("reasoning"):
                    print(f"  Reasoning: {evaluation['reasoning']}")
                print()
            
            history.append({
                "user": user_input,
                "agent": response
            })
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MindTheGoal - Goal Success Rate evaluation for multi-turn conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate MultiWOZ with 50 samples
  python main.py evaluate --dataset multiwoz --sample 50

  # Evaluate SGD with full dataset
  python main.py evaluate --dataset sgd

  # Resume an interrupted evaluation
  python main.py evaluate --dataset custom --data-dir data/multiwoz --sample 100 --resume

  # Start the API server
  python main.py server

  # Interactive chat with evaluation
  python main.py chat --domain restaurant
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a dataset")
    eval_parser.add_argument(
        "--dataset", "-d",
        required=True,
        choices=["multiwoz", "sgd", "custom"],
        help="Dataset to evaluate"
    )
    eval_parser.add_argument(
        "--sample", "-s",
        type=int,
        default=None,
        help="Number of dialogues to sample (default: full dataset)"
    )
    eval_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    eval_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    eval_parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory for custom datasets"
    )
    eval_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start the API server")
    server_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)"
    )
    server_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to bind (default: 8000)"
    )
    server_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with evaluation")
    chat_parser.add_argument(
        "--domain",
        choices=["restaurant", "hotel", "train", "taxi", "attraction"],
        default=None,
        help="Task domain for the chat agent"
    )
    chat_parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    if args.command == "evaluate":
        asyncio.run(run_evaluation(
            dataset=args.dataset,
            sample_size=args.sample,
            random_seed=args.seed,
            output_file=args.output,
            data_dir=getattr(args, 'data_dir', None),
            resume=getattr(args, 'resume', False)
        ))
    
    elif args.command == "server":
        run_server(
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    
    elif args.command == "chat":
        asyncio.run(run_chat(
            domain=args.domain,
            evaluate=not args.no_eval
        ))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
