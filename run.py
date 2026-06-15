import sys
from app.ui import create_ui

def main():
    try:
        # Create and launch UI
        demo = create_ui()
        
        # Configure launch (bind to all interfaces for flexibility in server/local environments)
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
    except KeyboardInterrupt:
        print("\nShutting down PDF Chatbot server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error launching server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
