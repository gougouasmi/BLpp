#ifndef MESSAGE_QUEUE_HPP
#define MESSAGE_QUEUE_HPP

#include <mutex>
#include <queue>
#include <thread>

template <class T> class MessageQueue {
public:
  void send(T &&msg) {
    std::lock_guard<std::mutex> lck(_mtx);
    _messages.push(std::move(msg));
    _cond.notify_one();
  };
  T fetch() {
    std::unique_lock<std::mutex> ulock(_mtx);
    _cond.wait(ulock, [this]() { return !_messages.empty() || !active; });

    if (active) {
      T msg = _messages.front();
      _messages.pop();

      return msg;
    }

    return T::StopMessage();
  };
  void stop() {
    std::lock_guard<std::mutex> lck(_mtx);
    active = false;
    _cond.notify_all();
  }

private:
  std::mutex _mtx;
  std::condition_variable _cond;
  std::queue<T> _messages;

  bool active = true;
};

#endif