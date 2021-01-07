class StopWatch():

    def __init__(self):
        self.start_time = time.time()

    def give(self):
        time_diff = round(time.time() - self.start_time)
        hour = str(time_diff // 3600).zfill(2)
        minute = str((time_diff % 3600) // 60).zfill(2)
        second = str(time_diff % 60).zfill(2)  # Same as time_diff - (minutes * 60)
        
        return f'[{hour}:{minute}:{second}]'