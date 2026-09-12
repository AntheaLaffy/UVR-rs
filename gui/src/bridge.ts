export interface Bridge {
  core: { invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> };
  event: { listen<T>(name: string, callback: (event: { payload: T }) => void): Promise<() => void> };
}
