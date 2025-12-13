#!/bin/bash

# ============================================================================
# Grokking Multi-seed Experiment Queue Manager
# ============================================================================
# Usage:
#   ./run.sh add "<command>"  - Add task to queue
#   ./run.sh start            - Start processing queue
#   ./run.sh status           - Check queue and task status
#   ./run.sh stop             - Stop queue processing & clean GPU memory
#   ./run.sh log              - View current task log
#   ./run.sh gpu              - Check GPU usage
#   ./run.sh cleangpu         - Fast GPU cleanup (non-interactive)
#   ./run.sh clear            - Clear all queued tasks
#   ./run.sh killgpu          - Force kill ALL GPU processes (interactive)
# ============================================================================

LOG_DIR="logs"
QUEUE_DIR="queue"
QUEUE_FILE="$QUEUE_DIR/tasks.queue"
QUEUE_PID_FILE="$QUEUE_DIR/queue_monitor.pid"
CURRENT_TASK_PID="$QUEUE_DIR/current_task.pid"
CURRENT_TASK_LOG="$QUEUE_DIR/current_task.log"

mkdir -p $LOG_DIR
mkdir -p $QUEUE_DIR

# ============================================================================
# Function: Add Task to Queue
# ============================================================================
add_task() {
    if [ -z "$1" ]; then
        echo "Error: No command provided"
        echo "Usage: ./run.sh add \"<command>\""
        echo ""
        echo "Examples:"
        echo "  ./run.sh add \"python grokking_multiseed.py --batch_size 512\""
        echo "  ./run.sh add \"python grokking_multiseed.py --batch_size 256 --lr 5e-4\""
        exit 1
    fi
    
    echo "$1" >> $QUEUE_FILE
    QUEUE_SIZE=$(wc -l < $QUEUE_FILE 2>/dev/null || echo 0)
    echo "✓ Task added to queue (position: $QUEUE_SIZE)"
    echo "Command: $1"
}

# ============================================================================
# Function: Process Next Task
# ============================================================================
process_next_task() {
    # Check if queue is empty
    if [ ! -f "$QUEUE_FILE" ] || [ ! -s "$QUEUE_FILE" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Queue empty. Exiting monitor."
        rm -f $QUEUE_PID_FILE
        return 1
    fi
    
    # Get first task from queue
    TASK=$(head -n 1 "$QUEUE_FILE")
    
    if [ -z "$TASK" ]; then
        return 1
    fi
    
    # Remove first line from queue
    tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp"
    mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
    
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$LOG_DIR/task_$TIMESTAMP.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting task: $TASK" | tee -a "$CURRENT_TASK_LOG"
    echo "Log: $LOG_FILE" | tee -a "$CURRENT_TASK_LOG"
    
    # Execute task in background
    eval "nohup $TASK > $LOG_FILE 2>&1 &"
    TASK_PID=$!
    echo $TASK_PID > $CURRENT_TASK_PID
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task started with PID: $TASK_PID" | tee -a "$CURRENT_TASK_LOG"
    
    # Wait for task to complete
    while ps -p $TASK_PID > /dev/null 2>&1; do
        sleep 10
    done
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task completed (PID: $TASK_PID)" | tee -a "$CURRENT_TASK_LOG"
    rm -f $CURRENT_TASK_PID
    
    return 0
}

# ============================================================================
# Function: Start Queue Monitor
# ============================================================================
start_queue() {
    # Check if monitor is already running
    if [ -f "$QUEUE_PID_FILE" ]; then
        MONITOR_PID=$(cat $QUEUE_PID_FILE)
        if ps -p $MONITOR_PID > /dev/null 2>&1; then
            echo "Queue monitor already running (PID: $MONITOR_PID)"
            exit 0
        fi
    fi
    
    # Check if queue has tasks
    if [ ! -f "$QUEUE_FILE" ] || [ ! -s "$QUEUE_FILE" ]; then
        echo "Queue is empty. Add tasks first:"
        echo "  ./run.sh add \"<command>\""
        exit 1
    fi
    
    QUEUE_SIZE=$(wc -l < $QUEUE_FILE)
    echo "=========================================="
    echo "Starting Queue Monitor"
    echo "Tasks in queue: $QUEUE_SIZE"
    echo "=========================================="
    
    # Start monitor in background
    (
        > $CURRENT_TASK_LOG  # Clear log
        while process_next_task; do
            sleep 2
        done
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] All tasks completed!" | tee -a "$CURRENT_TASK_LOG"
    ) &
    
    MONITOR_PID=$!
    echo $MONITOR_PID > $QUEUE_PID_FILE
    
    echo "Queue monitor started (PID: $MONITOR_PID)"
    echo ""
    echo "Commands:"
    echo "  ./run.sh status   - Check progress"
    echo "  ./run.sh log      - View current task log"
    echo "  ./run.sh stop     - Stop queue"
}

# ============================================================================
# Function: View Current Log
# ============================================================================
view_log() {
    if [ -f "$CURRENT_TASK_PID" ]; then
        TASK_PID=$(cat $CURRENT_TASK_PID)
        # Find the most recent log file
        LATEST_LOG=$(ls -t $LOG_DIR/task_*.log 2>/dev/null | head -1)
        
        if [ -n "$LATEST_LOG" ]; then
            echo "Viewing current task log: $LATEST_LOG"
            echo "Press Ctrl+C to exit"
            echo "=========================================="
            tail -f "$LATEST_LOG"
        else
            echo "No active task log found"
        fi
    else
        echo "No task currently running"
        echo ""
        echo "Recent logs:"
        ls -lth $LOG_DIR/task_*.log 2>/dev/null | head -5
    fi
}

# ============================================================================
# Function: Check Status
# ============================================================================
check_status() {
    echo "=========================================="
    echo "Queue Status"
    echo "=========================================="
    
    # Check monitor status
    if [ -f "$QUEUE_PID_FILE" ]; then
        MONITOR_PID=$(cat $QUEUE_PID_FILE)
        if ps -p $MONITOR_PID > /dev/null 2>&1; then
            echo "Monitor: RUNNING ✓ (PID: $MONITOR_PID)"
        else
            echo "Monitor: STOPPED ✗"
            rm -f $QUEUE_PID_FILE
        fi
    else
        echo "Monitor: NOT STARTED"
    fi
    
    echo ""
    echo "----------------------------------------"
    echo "Current Task:"
    echo "----------------------------------------"
    
    # Check current task
    if [ -f "$CURRENT_TASK_PID" ]; then
        TASK_PID=$(cat $CURRENT_TASK_PID)
        if ps -p $TASK_PID > /dev/null 2>&1; then
            echo "Status: RUNNING ✓ (PID: $TASK_PID)"
            ps -p $TASK_PID -o pid,ppid,%cpu,%mem,etime,cmd
        else
            echo "Status: Task PID exists but process not found"
        fi
    else
        echo "Status: No task running"
    fi
    
    echo ""
    echo "----------------------------------------"
    echo "Queued Tasks:"
    echo "----------------------------------------"
    
    if [ -f "$QUEUE_FILE" ] && [ -s "$QUEUE_FILE" ]; then
        QUEUE_SIZE=$(wc -l < $QUEUE_FILE)
        echo "Tasks in queue: $QUEUE_SIZE"
        echo ""
        nl -w2 -s'. ' "$QUEUE_FILE"
    else
        echo "Queue is empty"
    fi
    
    echo ""
    echo "----------------------------------------"
    echo "Recent Activity:"
    echo "----------------------------------------"
    if [ -f "$CURRENT_TASK_LOG" ]; then
        tail -10 "$CURRENT_TASK_LOG"
    else
        echo "No activity log"
    fi
}

# ============================================================================
# Function: Stop Queue (Enhanced GPU Memory Cleanup)
# ============================================================================
stop_queue() {
    echo "=========================================="
    echo "Stopping Queue & Cleaning GPU Memory"
    echo "=========================================="
    
    # Stop current task
    if [ -f "$CURRENT_TASK_PID" ]; then
        TASK_PID=$(cat $CURRENT_TASK_PID)
        if ps -p $TASK_PID > /dev/null 2>&1; then
            echo "Stopping current task (PID: $TASK_PID)..."
            kill $TASK_PID 2>/dev/null
            sleep 2
            if ps -p $TASK_PID > /dev/null 2>&1; then
                kill -9 $TASK_PID 2>/dev/null
            fi
            echo "✓ Task stopped"
        fi
        rm -f $CURRENT_TASK_PID
    fi
    
    # Stop monitor
    if [ -f "$QUEUE_PID_FILE" ]; then
        MONITOR_PID=$(cat $QUEUE_PID_FILE)
        if ps -p $MONITOR_PID > /dev/null 2>&1; then
            echo "Stopping queue monitor (PID: $MONITOR_PID)..."
            kill $MONITOR_PID 2>/dev/null
            sleep 1
            if ps -p $MONITOR_PID > /dev/null 2>&1; then
                kill -9 $MONITOR_PID 2>/dev/null
            fi
            echo "✓ Monitor stopped"
        fi
        rm -f $QUEUE_PID_FILE
    fi
    
    # Kill ALL grokking_multiseed.py related processes
    echo ""
    echo "Cleaning all related Python processes..."
    PIDS=$(ps aux | grep "grokking_multiseed.py" | grep -v grep | awk '{print $2}')
    if [ -n "$PIDS" ]; then
        echo "Found PIDs: $PIDS"
        for pid in $PIDS; do
            echo "  Killing PID $pid..."
            kill -9 $pid 2>/dev/null
        done
        sleep 1
    fi
    
    # Kill any remaining Python processes using CUDA
    echo ""
    echo "Cleaning Python processes with GPU memory..."
    GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    if [ -n "$GPU_PIDS" ]; then
        for gpu_pid in $GPU_PIDS; do
            PROCESS_NAME=$(ps -p $gpu_pid -o comm= 2>/dev/null)
            if [[ "$PROCESS_NAME" == "python"* ]]; then
                echo "  Killing GPU process PID $gpu_pid ($PROCESS_NAME)..."
                kill -9 $gpu_pid 2>/dev/null
            fi
        done
        sleep 2
    fi
    
    # Deep clean: Kill all processes using NVIDIA devices
    echo ""
    echo "Deep cleaning NVIDIA device users..."
    sudo fuser -k /dev/nvidia* >/dev/null 2>&1 || true
    sleep 2
    
    echo ""
    echo "✓ Queue stopped and GPU memory cleaned"
    echo ""
    echo "Verifying GPU memory status..."
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
    echo ""
    echo "💡 If GPU memory is still occupied, run: ./run.sh killgpu"
}

# ============================================================================
# Function: Clear Queue
# ============================================================================
clear_queue() {
    if [ -f "$QUEUE_FILE" ]; then
        QUEUE_SIZE=$(wc -l < $QUEUE_FILE)
        > $QUEUE_FILE  # Clear file
        echo "✓ Cleared $QUEUE_SIZE tasks from queue"
    else
        echo "Queue is already empty"
    fi
}

# ============================================================================
# Function: Check GPU
# ============================================================================
check_gpu() {
    echo "=========================================="
    echo "GPU Status"
    echo "=========================================="
    nvidia-smi
}

# ============================================================================
# Function: Kill All GPU Processes (Force cleanup)
# ============================================================================
killgpu() {
    echo "=========================================="
    echo "Force Kill ALL GPU Processes"
    echo "=========================================="
    echo "⚠️  WARNING: This will kill ALL Python processes using GPU!"
    echo ""
    
    # Show current GPU processes
    echo "Current GPU processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    echo ""
    
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        return
    fi
    
    # Get all PIDs using GPU
    GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    
    if [ -z "$GPU_PIDS" ]; then
        echo "No GPU processes found."
        return
    fi
    
    echo "Killing processes..."
    for pid in $GPU_PIDS; do
        PROCESS_NAME=$(ps -p $pid -o comm= 2>/dev/null || echo "unknown")
        echo "  Killing PID $pid ($PROCESS_NAME)..."
        kill -9 $pid 2>/dev/null
    done
    
    sleep 2
    
    echo ""
    echo "✓ GPU cleanup complete"
    echo ""
    echo "Current GPU memory status:"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
}

# ============================================================================
# Main Script
# ============================================================================

case "$1" in
    add)
        shift
        add_task "$*"
        ;;
    start)
        start_queue
        ;;
    log)
        view_log
        ;;
    status)
        check_status
        ;;
    stop)
        stop_queue
        ;;
    clear)
        clear_queue
        ;;
    gpu)
        check_gpu
        ;;
    killgpu)
        killgpu
        ;;
    cleangpu)
        # Non-interactive fast GPU cleanup
        echo "=========================================="
        echo "Fast GPU Cleanup (Non-interactive)"
        echo "=========================================="
        echo "Killing all GPU processes..."
        sudo fuser -k /dev/nvidia* >/dev/null 2>&1 || true
        sleep 2
        echo "✓ GPU cleanup complete"
        nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
        ;;
    *)
        echo "Usage: $0 {add|start|status|stop|log|clear|gpu|killgpu|cleangpu}"
        echo ""
        echo "Commands:"
        echo "  add \"<cmd>\"  - Add task to queue"
        echo "  start        - Start processing queue"
        echo "  status       - Check queue and task status"
        echo "  stop         - Stop queue processing & clean GPU"
        echo "  log          - View current task log"
        echo "  clear        - Clear all queued tasks"
        echo "  gpu          - Check GPU usage"
        echo "  cleangpu     - Fast GPU memory cleanup (non-interactive)"
        echo "  killgpu      - Force kill ALL GPU processes (interactive)"
        echo ""
        echo "Example workflow:"
        echo "  1. ./run.sh add \"python grokking_multiseed.py --batch_size 512\""
        echo "  2. ./run.sh add \"python grokking_multiseed.py --batch_size 256\""
        echo "  3. ./run.sh start"
        echo "  4. ./run.sh status"
        echo ""
        echo "GPU memory cleanup (if stop doesn't fully clean):"
        echo "  ./run.sh cleangpu  - Quick non-interactive cleanup (recommended)"
        echo "  ./run.sh killgpu   - Interactive cleanup with confirmation"
        exit 1
        ;;
esac
