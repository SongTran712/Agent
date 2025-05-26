"use client";
import Image from "next/image";
import { useState, useEffect, useRef } from "react";
import ReactMarkdown from 'react-markdown'
import { marked } from 'marked';
import { v4 as uuidv4 } from 'uuid';
import { FaSearch, FaChevronLeft, FaChevronRight } from 'react-icons/fa';
import { useRouter } from 'next/navigation'
import DOMPurify from 'dompurify';

export default function Home() {
  const router = useRouter();
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [inputColor, setInputColor] = useState('black');
  const [loading, setLoading] = useState(false);
  const [sessionID, setSessionID] = useState(null);
  const [showSidebar, setShowSidebar] = useState(true);
  const [collapsed, setCollapsed] = useState(false);
  const [image, setImage] = useState(null);
  const bottomRef = useRef(null);
  const [loadMessage, setLoadMessage] = useState(false)
  const handleDrop = async (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    setMessages([])
    if (file && file.type.startsWith("image/")) {

      const reader = new FileReader();
      reader.onload = () => setImage(reader.result);  // for preview
      reader.readAsDataURL(file);
      await uploadToBackend(file);
    }
  };

  const handleChange = async (e) => {
    const file = e.target.files[0];
    setMessages([])
    if (file && file.type.startsWith("image/")) {

      const reader = new FileReader();
      reader.onload = () => setImage(reader.result);
      reader.readAsDataURL(file);
      await uploadToBackend(file);
    }
  };

  const uploadToBackend = async (file) => {
    const formData = new FormData();
    formData.append("file", file);  // ✅ KEY: must be named 'file'

    try {
      setLoadMessage(true)
      const response = await fetch("http://192.168.30.172:8000/api/image", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to fetch response from the server');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      let botMessage = { role: "assistant", content: "" }; // Initial empty AI message
      setMessages((prev) => [...prev, botMessage]); // Add empty message to array

      const readStream = async () => {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const token = decoder.decode(value, { stream: true });

          setMessages((prev) => {
            let updated = [...prev];
            updated[updated.length - 1].content += token;
            return updated;
          });
        }
        setLoading(false);
        setLoadMessage(false)
      };

      // Start reading the stream
      readStream();

    } catch (error) {
      console.error("Error occurred while sending the message:", error);
      setLoading(false); // Ensure loading stops if there is an error
      // Optionally show an error message to the user
      setMessages((prev) => [...prev, { role: "assistant", content: "Sorry, something went wrong!" }]);
    }
  };

  const handlePaste = (e) => {
    const item = Array.from(e.clipboardData.items).find(i => i.type.includes("image"));
    if (item) {
      const file = item.getAsFile();
      setImage(URL.createObjectURL(file));
    }
  };

  // const handleChange = (e) => {
  //   const file = e.target.files[0];
  //   if (file && file.type.startsWith("image/")) {
  //     setImage(URL.createObjectURL(file));
  //   }
  // };

  const ChatItem = ({ imgSrc, name, lastMsg, route }) => (
    <div
      className="flex flex-row py-4 px-4 items-center border-b hover:bg-gray-100 transition cursor-pointer"
      onClick={() => router.push(route)}
    >
      <div className="w-1/4">
        <img src={imgSrc} alt={name} className="object-cover h-12 w-12 rounded-full" />
      </div>
      <div className="w-full ml-4">
        <div className="text-md text-black font-semibold">{name}</div>
        <span className="text-gray-500 text-sm truncate">{lastMsg}</span>
      </div>
    </div>
  );

  const chatList = [
    { imgSrc: '/agent.avif', name: 'Agent', lastMsg: 'Last message preview', route: '/' },
    { imgSrc: '/vlm.jpg', name: 'VLMs', lastMsg: 'Last message preview', route: '/vlm' },
  ];

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  useEffect(() => {
    if (messages.length === 0 && !sessionID) {
      const generatedSessionID = uuidv4();
      setSessionID(generatedSessionID);
    }
  }, [messages, sessionID]);



  return (
    <div className="flex flex-col h-screen">
      <header style={{ backgroundColor: "#09195d", position: "fixed", top: 0, width: "100%", height: "80px" }} className="text-white py-4 px-8 shadow-md relative flex items-center">
        <div className="flex items-center gap-4">
          <Image src="/veron-logo-rm.png" alt="Veron Logo" width={160} height={40} />
        </div>
        <div className="absolute left-1/2 transform -translate-x-1/2 text-3xl font-bold tracking-wide flex items-center gap-4">
          <h1>VERON Chatbot</h1>
          <Image src="/robo.png" alt="Robot Logo" width={80} height={10} />
        </div>
        <span className="ml-auto text-gray-300 text-sm">Powered by VERON R&D</span>
      </header>

      <main className="flex-1 w-full flex flex-col gap-8 row-start-2 pt-20 min-h-0">
        <div className="w-full flex flex-col flex-1 min-h-0">
          <div className="flex flex-row justify-between bg-white h-full">
            <div className={`flex flex-col ${collapsed ? 'w-16' : 'w-1/6'} border-r-2 transition-all duration-300 overflow-y-auto`}>
              <div className="flex justify-between border-b-2 py-4 px-2">
                {!collapsed && (
                  <div className="flex items-center border-2 border-gray-200 rounded-2xl px-2">
                    <input
                      type="text"
                      placeholder="Search chatting"
                      className="py-2 px-2 w-full focus:outline-none rounded-2xl"
                    />
                    <FaSearch className="text-gray-500 h-4 w-4 ml-2" />
                  </div>
                )}
                <div className="flex justify-end p-2">
                  <button onClick={() => setCollapsed(!collapsed)}>
                    {collapsed ? <FaChevronRight className="text-gray-600" /> : <FaChevronLeft className="text-gray-600" />}
                  </button>
                </div>
              </div>
              {!collapsed && (
                <div>
                  {chatList.map((chat, idx) => (
                    <ChatItem key={idx} {...chat} />
                  ))}
                </div>
              )}
            </div>

            <div className="w-full flex flex-col items-center bg-cover bg-center overflow-hidden flex-1 min-h-0">
              {/* <div style={{ backgroundImage: "url('/background1.jpg')" }} className="w-full flex flex-col justify-center items-center bg-cover bg-center overflow-hidden flex-1 min-h-0"> */}
              <h1 className="text-3xl font-bold text-[#09195d] mt-10 mb-10">Agriculture assistant</h1>
              <div className="w-full max-w-5xl mx-auto flex flex-row justify-center items-start gap-20">
                {/* Left: Image Upload Box */}
                <div
                  className="flex flex-col w-full max-w-xl h-[60vh] p-6 border-2 border-dashed border-[#09195d] rounded-lg text-center text-gray-500"
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  onPaste={handlePaste}
                >
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleChange}
                    className="hidden"
                    id="fileUpload"
                  />
                  <label htmlFor="fileUpload" className="cursor-pointer block">
                    <p className="mb-2 text-lg font-semibold text-[#09195d]">
                      Drag & Drop, Paste, or Click to Upload
                    </p>
                    <p className="text-sm text-gray-400">Only image files are supported.</p>
                  </label>

                  {image && (
                    <div className="mt-4 flex-1 overflow-hidden">
                      <img src={image} alt="Uploaded" className="w-full h-full object-contain rounded shadow" />
                    </div>
                  )}
                </div>

                {/* Right: Assistant's Response Box */}
                <div className="flex flex-col h-[60vh] w-full max-w-2xl p-4 bg-white rounded border border-gray-300 shadow text-left text-gray-800">
                  <h2 className="font-semibold text-[#09195d] mb-2">Assistant's Response:</h2>
                  <p className="flex-1 overflow-y-auto whitespace-pre-line">
                    {loadMessage ? (
                      <div className="flex items-center justify-center h-full">
                        <div className="w-8 h-8 border-4 border-gray-300 border-t-blue-500 rounded-full animate-spin" />
                      </div>
                    ) : (
                      messages.length > 0 && messages[messages.length - 1]?.content
                    )}
                  </p>
                </div>
              </div>


            </div>
          </div>
        </div>
      </main>
    </div>
  );
}