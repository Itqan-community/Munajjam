"""
        """
        pass


    def transcribe_stream(self, audio_path: str | Path) -> Iterator[Segment]:
        """
        Transcribe audio and yield segments as they are processed.


        Default implementation transcribes all at once and yields.
        Override for true streaming support.


        Args:
            audio_path: Path to the audio file


        Yields:
            Segment objects as they are transcribed
        """
        segments = self.transcribe(audio_path)
        yield from segments


    async def transcribe_stream_async(self, audio_path: str | Path) -> AsyncIterator[Segment]:
        """
        Asynchronously transcribe and yield segments.


        Default implementation transcribes all at once and yields.
        Override for true streaming support.


        Args:
            audio_path: Path to the audio file


        Yields:
            Segment objects as they are transcribed
        """
        segments = await self.transcribe_async(audio_path)
        for segment in segments:
            yield segment


    @abstractmethod
    def load(self) -> None:
        """
        Load the model into memory.


        Call this before transcription to pre-load the model.
        Useful for avoiding cold start latency.
        """
        pass


    @abstractmethod
    def unload(self) -> None:
        """
        Unload the model from memory.


        Call this to free up memory when done transcribing.
        """
        pass


    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        pass


    def __enter__(self) -> "BaseTranscriber":
        """Context manager entry - loads the model."""
        self.load()
        return self


    def __exit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object | None
    ) -> None:
        """Context manager exit - unloads the model."""
        self.unload()

