import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [openaiKey, setOpenaiKey] = useState('');
  const [pineconeKey, setPineconeKey] = useState('');
  const [pineconeEnv, setPineconeEnv] = useState('');
  const [urls, setUrls] = useState('');
  const [store, setStore] = useState('FAISS');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  const handleSetApiKey = async () => {
    setLoading(true);
    try {
      const response = await axios.post('/set-api-key', {
        openai_key: openaiKey,
        pinecone_key: pineconeKey,
        pinecone_env: pineconeEnv
      });
      setMessage(response.data.message);
    } catch (error) {
      setMessage('Error: ' + error.response.data.detail);
    }
    setLoading(false);
  };

  const handleTrainModel = async () => {
    setLoading(true);
    try {
      const response = await axios.post('/train', {
        urls: urls.split('\n').filter(url => url.trim() !== ''),
        store: store
      });
      setMessage(response.data.message);
    } catch (error) {
      setMessage('Error: ' + error.response.data.detail);
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <h1>Admin Interface</h1>

      <div>
        <h2>Set API Keys</h2>
        <div>
          <label htmlFor="openai_key">OpenAI API Key:</label>
          <input type="text" id="openai_key" value={openaiKey} onChange={(e) => setOpenaiKey(e.target.value)} />
        </div>
        <div>
          <label htmlFor="pinecone_key">Pinecone API Key:</label>
          <input type="text" id="pinecone_key" value={pineconeKey} onChange={(e) => setPineconeKey(e.target.value)} />
        </div>
        <div>
          <label htmlFor="pinecone_env">Pinecone API Environment:</label>
          <input type="text" id="pinecone_env" value={pineconeEnv} onChange={(e) => setPineconeEnv(e.target.value)} />
        </div>
        <button onClick={handleSetApiKey} disabled={loading}>Set API Keys</button>
      </div>

      <div>
        <h2>Train Chatbot</h2>
        <div>
          <label htmlFor="urls">Enter URLs (one per line):</label>
          <textarea id="urls" value={urls} onChange={(e) => setUrls(e.target.value)} />
        </div>
        <div>
          <label htmlFor="store">Store:</label>
          <select id="store" value={store} onChange={(e) => setStore(e.target.value)}>
            <option value="FAISS">FAISS</option>
            <option value="Pinecone">Pinecone</option>
          </select>
        </div>
        <button onClick={handleTrainModel} disabled={loading}>Train Chatbot</button>
      </div>

      {message && <div>{message}</div>}
    </div>
  );
}

export default App;
