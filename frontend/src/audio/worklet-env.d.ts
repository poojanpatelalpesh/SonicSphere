// Minimal typings for AudioWorkletGlobalScope (not included in lib.dom).
declare const sampleRate: number;
declare const currentFrame: number;
declare const currentTime: number;

declare class AudioWorkletProcessor {
  readonly port: MessagePort;
  constructor(options?: unknown);
}

declare function registerProcessor(name: string, ctor: new (options?: unknown) => AudioWorkletProcessor): void;
