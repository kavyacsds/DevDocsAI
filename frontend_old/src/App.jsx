import { useState } from "react";


function App() {

  const [file, setFile] = useState(null);

  const [question, setQuestion] = useState("");

  const [messages, setMessages] = useState([]);

  const [uploadMessage, setUploadMessage] = useState("");

  const [loading, setLoading] = useState(false);


  // ==================================================
  // UPLOAD PDF
  // ==================================================

  const handleUpload = async () => {

    if (!file) {

      setUploadMessage(
        "Please select a PDF first."
      );

      return;
    }


    const formData = new FormData();

    formData.append(
      "file",
      file
    );


    setUploadMessage(
      "Uploading and indexing..."
    );


    try {

      const response = await fetch(
        "http://127.0.0.1:8000/upload",
        {
          method: "POST",
          body: formData
        }
      );


      const data = await response.json();


      if (!response.ok) {

        setUploadMessage(
          data.detail || "Upload failed."
        );

        return;
      }


      setUploadMessage(
        `${data.filename} uploaded successfully. ` +
        `${data.chunks_added} chunks added.`
      );

    } catch (error) {

      setUploadMessage(
        "Could not connect to the backend."
      );
    }
  };


  // ==================================================
  // ASK QUESTION
  // ==================================================

  const handleChat = async () => {

    const trimmedQuestion =
      question.trim();


    if (!trimmedQuestion || loading) {
      return;
    }


    // Add user message

    const userMessage = {
      role: "user",
      content: trimmedQuestion
    };


    setMessages(
      previous => [
        ...previous,
        userMessage
      ]
    );


    setQuestion("");

    setLoading(true);


    try {

      const response = await fetch(
        "http://127.0.0.1:8000/chat",
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json"
          },

          body: JSON.stringify({
    question: trimmedQuestion,

    chat_history: messages.map(message => ({
        role: message.role,
        content: message.content
    }))
})
        }
      );


      const data = await response.json();


      if (!response.ok) {

        setMessages(
          previous => [
            ...previous,
            {
              role: "assistant",
              content:
                data.detail ||
                "Something went wrong."
            }
          ]
        );

        return;
      }


      // Add AI message

      setMessages(
        previous => [
          ...previous,
          {
            role: "assistant",
            content: data.answer,
            sources: data.sources || []
          }
        ]
      );

    } catch (error) {

      setMessages(
        previous => [
          ...previous,
          {
            role: "assistant",
            content:
              "Could not connect to the backend."
          }
        ]
      );

    } finally {

      setLoading(false);
    }
  };


  // ==================================================
  // ENTER KEY
  // ==================================================

  const handleKeyDown = (event) => {

    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {

      event.preventDefault();

      handleChat();
    }
  };


  return (

    <div className="app">

      <h1>DevDocs AI</h1>


      {/* ========================================== */}
      {/* UPLOAD */}
      {/* ========================================== */}

      <div className="upload-section">

        <h2>Upload PDF</h2>

        <input
          type="file"
          accept=".pdf"
          onChange={(event) =>
            setFile(
              event.target.files[0]
            )
          }
        />

        <button
          onClick={handleUpload}
        >
          Upload
        </button>

        {uploadMessage && (
          <p>{uploadMessage}</p>
        )}

      </div>


      {/* ========================================== */}
      {/* CHAT */}
      {/* ========================================== */}

      <div className="chat-section">

        <h2>Ask your documents</h2>


        {/* MESSAGE HISTORY */}

        <div className="messages">

          {messages.length === 0 && (

            <p>
              Ask a question about your uploaded
              documents.
            </p>

          )}


          {messages.map(
            (message, index) => (

              <div
                key={index}
                className={
                  message.role === "user"
                    ? "message user-message"
                    : "message assistant-message"
                }
              >

                <strong>
                  {message.role === "user"
                    ? "You"
                    : "DevDocs AI"}
                </strong>


                <p>
                  {message.content}
                </p>


                {/* SOURCES */}

                {message.sources &&
                  message.sources.length > 0 && (

                    <div>

                      <strong>
                        Sources
                      </strong>

                      <ul>

                        {message.sources.map(
                          (source, sourceIndex) => (

                            <li
                              key={sourceIndex}
                            >

                              {source.file}
                              {" — Page "}
                              {source.page}

                            </li>

                          )
                        )}

                      </ul>

                    </div>

                  )}

              </div>

            )
          )}

        </div>


        {/* LOADING */}

        {loading && (

          <p>
            DevDocs AI is thinking...
          </p>

        )}


        {/* INPUT */}

        <div className="chat-input">

          <textarea
            placeholder="Ask a question..."
            value={question}
            onChange={(event) =>
              setQuestion(
                event.target.value
              )
            }
            onKeyDown={handleKeyDown}
            rows={3}
          />

          <button
            onClick={handleChat}
            disabled={loading}
          >
            {loading
              ? "Thinking..."
              : "Send"}
          </button>

        </div>

      </div>

    </div>
  );
}


export default App;