import time

class Timer:
    """
    Uma classe de utilitário para medir tempos de execução e 
    checkpoints nomeados.
    
    Usado para medir o tempo de inicialização de cada biblioteca.
    """
    def __init__(self):
        self.start_time = time.perf_counter()
        self.checkpoints = {}
        # Registra o início como o primeiro checkpoint
        self.checkpoints["_start"] = self.start_time

    def checkpoint(self, name):
        """
        Registra um checkpoint nomeado com o tempo desde o início.
        """
        t = time.perf_counter()
        self.checkpoints[name] = t
        
    def get_all_checkpoints(self):
        """
        Retorna um dicionário com os tempos de cada checkpoint 
        em relação ao checkpoint anterior.
        """
        checkpoint_times = {}
        last_time = self.start_time
        
        # Ignora o _start, itera em ordem de inserção
        for name, t in self.checkpoints.items():
            if name == "_start":
                continue
                
            elapsed = t - last_time
            checkpoint_times[name] = elapsed
            last_time = t
            
        return checkpoint_times

    def total(self):
        """
        Retorna o tempo total desde a inicialização.
        """
        return time.perf_counter() - self.start_time