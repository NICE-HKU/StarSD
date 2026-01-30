#!/bin/bash

# ============================================================
# Script to start multiple Auto Base Clients
# 
# Features:
#   1. Start specified number of clients, each using different CUDA device
#   2. Record all process PIDs to file for easy termination
#   3. Provide one-command stop for all clients
#
# Usage:
#   Start clients:  ./start_three_clients.sh
#   Stop clients:   ./start_three_clients.sh stop
#   Show status:    ./start_three_clients.sh status
# ============================================================

# Configuration - modify as needed
CUDA_DEVICES=(3)        # List of CUDA devices to use
SCRIPT_NAME="framework/auto_base_client.py"
PID_FILE=".client_pids"     # File to store PIDs
STARTUP_DELAY=0.5           # Delay between client startups (seconds)

# Extra arguments (optional)
EXTRA_ARGS=""               # e.g.: "--mode sequential --questions 100"

# ============================================================
# Function definitions
# ============================================================

start_clients() {
    echo "Starting ${#CUDA_DEVICES[@]} Auto Base Clients..."
    echo ""
    
    # Clear or create PID file
    > "$PID_FILE"
    
    for i in "${!CUDA_DEVICES[@]}"; do
        cuda_id=${CUDA_DEVICES[$i]}
        client_num=$((i + 1))
        
        echo "[$client_num/${#CUDA_DEVICES[@]}] Starting client (cuda:$cuda_id)..."
        
        # Start client and get PID
        python "$SCRIPT_NAME" --cuda "$cuda_id" --client_id "$i" $EXTRA_ARGS &
        pid=$!
        
        # Record PID
        echo "$pid" >> "$PID_FILE"
        echo "    PID: $pid"
        
        # Wait before starting next client
        if [ $i -lt $((${#CUDA_DEVICES[@]} - 1)) ]; then
            sleep "$STARTUP_DELAY"
        fi
    done
    
    echo ""
    echo "All clients started successfully!"
    echo ""
    echo "PID list saved to: $PID_FILE"
    echo "   Contents: $(cat $PID_FILE | tr '\n' ' ')"
    echo ""
    echo "To stop all clients: $0 stop"
    echo "To check status:     $0 status"
}

stop_clients() {
    if [ ! -f "$PID_FILE" ]; then
        echo "PID file not found: $PID_FILE"
        echo "   No running clients, or clients were started differently"
        echo ""
        echo "Attempting to terminate by process name..."
        pkill -f "$SCRIPT_NAME" && echo "Termination signal sent" || echo "No matching processes found"
        return
    fi
    
    echo "Stopping all clients..."
    echo ""
    
    count=0
    while read -r pid; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "   Terminating process PID: $pid"
            kill "$pid" 2>/dev/null
            count=$((count + 1))
        else
            echo "   Process PID: $pid no longer exists"
        fi
    done < "$PID_FILE"
    
    # Wait for processes to end
    sleep 1
    
    # Force kill any remaining processes
    while read -r pid; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "   Force killing process PID: $pid"
            kill -9 "$pid" 2>/dev/null
        fi
    done < "$PID_FILE"
    
    # Clean up PID file
    rm -f "$PID_FILE"
    
    echo ""
    echo "Terminated $count client process(es)"
}

show_status() {
    echo "Client Status"
    echo "=============================================="
    
    if [ ! -f "$PID_FILE" ]; then
        echo "PID file not found: $PID_FILE"
        echo ""
        echo "Searching for related processes..."
        pgrep -af "$SCRIPT_NAME" || echo "   No running clients found"
        return
    fi
    
    echo ""
    running=0
    stopped=0
    
    while read -r pid; do
        if [ -n "$pid" ]; then
            if kill -0 "$pid" 2>/dev/null; then
                # Get process info
                cmd=$(ps -p "$pid" -o args= 2>/dev/null | head -c 80)
                echo "   [RUNNING] PID $pid"
                echo "      Command: $cmd"
                running=$((running + 1))
            else
                echo "   [STOPPED] PID $pid"
                stopped=$((stopped + 1))
            fi
        fi
    done < "$PID_FILE"
    
    echo ""
    echo "Summary: $running running, $stopped stopped"
}

# ============================================================
# Main program
# ============================================================

case "${1:-start}" in
    start)
        start_clients
        ;;
    stop|kill)
        stop_clients
        ;;
    status|ps)
        show_status
        ;;
    restart)
        stop_clients
        sleep 1
        start_clients
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        echo ""
        echo "  start   - Start all clients (default)"
        echo "  stop    - Stop all clients"
        echo "  status  - Show running status"
        echo "  restart - Restart all clients"
        exit 1
        ;;
esac
