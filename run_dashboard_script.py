#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para executar o dashboard Streamlit.

Uso:
    python scripts/run_dashboard.py [--port 8501] [--dev]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """Executa o dashboard Streamlit."""
    
    parser = argparse.ArgumentParser(description='Executar dashboard Streamlit')
    parser.add_argument('--port', type=int, default=8501,
                       help='Porta para o servidor (padrão: 8501)')
    parser.add_argument('--dev', action='store_true',
                       help='Modo desenvolvimento (auto-reload)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host para o servidor (padrão: localhost)')
    
    args = parser.parse_args()
    
    # Caminho para o app Streamlit
    app_path = Path(__file__).parent.parent / 'src' / 'dashboard' / 'streamlit_app.py'
    
    if not app_path.exists():
        print(f"❌ Erro: Arquivo não encontrado: {app_path}")
        sys.exit(1)
    
    # Comando do Streamlit
    cmd = [
        'streamlit', 'run', str(app_path),
        '--server.port', str(args.port),
        '--server.address', args.host
    ]
    
    # Adicionar opções de desenvolvimento
    if args.dev:
        cmd.extend(['--server.runOnSave', 'true'])
        cmd.extend(['--server.fileWatcherType', 'auto'])
    
    print("🚀 Iniciando dashboard Streamlit...")
    print(f"📱 Acesse: http://{args.host}:{args.port}")
    print(f"🛠️ Modo desenvolvimento: {'Ativado' if args.dev else 'Desativado'}")
    print("-" * 50)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Dashboard encerrado pelo usuário")
    except Exception as e:
        print(f"❌ Erro ao executar dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()