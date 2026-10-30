import { useEffect, useRef } from 'react';

/**
 * Reusable polling hook.
 * Invokes callback at specified interval while isEnabled is true.
 * Ensures no overlapping asynchronous executions and cleanly cancels on unmount.
 */
export function usePolling(
  callback: () => Promise<void> | void,
  intervalMs: number = 2500,
  isEnabled: boolean = true
): void {
  const savedCallback = useRef(callback);
  const isExecutingRef = useRef(false);

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (!isEnabled || intervalMs <= 0) return;

    let isMounted = true;

    const tick = async () => {
      if (isExecutingRef.current) return;
      isExecutingRef.current = true;
      try {
        if (isMounted) {
          await savedCallback.current();
        }
      } catch (err) {
        // Polling errors handled gracefully within the callback
      } finally {
        isExecutingRef.current = false;
      }
    };

    // Immediate first tick on activation
    tick();

    const id = setInterval(tick, intervalMs);

    return () => {
      isMounted = false;
      clearInterval(id);
    };
  }, [intervalMs, isEnabled]);
}

export default usePolling;
