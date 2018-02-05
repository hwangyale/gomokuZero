from threading import Thread, RLock, Condition


lock = RLock()
condition = Condition(lock)
