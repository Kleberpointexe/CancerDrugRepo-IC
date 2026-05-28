import logging
import os


def setup_logger(exp_path: str) -> logging.Logger:
    # Cria (ou recupera) um logger com o nome "experimento"
    logger = logging.getLogger("experimento")
    
    # Define o nível mínimo de mensagens que o logger aceita (DEBUG = todos os níveis)
    logger.setLevel(logging.DEBUG)

    # Define o formato das mensagens: data/hora, nível (INFO, DEBUG...) e texto
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # -------------------------------------------------------
    # HANDLER DE ARQUIVO
    # Salva todos os logs (DEBUG+) em "experiment.log"
    # dentro da pasta do experimento
    # -------------------------------------------------------
    log_file = os.path.join(exp_path, "experiment.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)      # Registra tudo no arquivo
    file_handler.setFormatter(formatter)      # Aplica o formato definido acima

    # -------------------------------------------------------
    # HANDLER DE CONSOLE
    # Exibe logs no terminal, mas apenas nível INFO ou acima
    # (ignora mensagens DEBUG para não poluir a saída)
    # -------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)    # Exibe apenas INFO, WARNING, ERROR, CRITICAL
    console_handler.setFormatter(formatter)   # Mesmo formato do arquivo

    # Registra os dois handlers no logger
    # A partir daqui, cada logger.info(...) ou logger.debug(...)
    # será direcionado para ambos os destinos
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger