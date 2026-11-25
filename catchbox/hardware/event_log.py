"""Non-blocking JSON-lines logging for Python ROS nodes."""
import json, queue, threading, time
from pathlib import Path

class EventLog:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.queue=queue.Queue(maxsize=4096);self.dropped=0
        self.file=self.path.open('a', buffering=1)
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
    def write(self,event,**fields):
        row=dict(event=event,wall_time_s=time.time(),monotonic_s=time.monotonic(),**fields)
        try:self.queue.put_nowait(row)
        except queue.Full:self.dropped+=1
    def _run(self):
        while True:
            row=self.queue.get()
            if row is None:break
            self.file.write(json.dumps(row,allow_nan=False)+'\n')
        self.file.close()
    def close(self):
        self.queue.put(None);self.thread.join(timeout=5)
