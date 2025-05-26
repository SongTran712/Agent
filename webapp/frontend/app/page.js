"use client";
import Image from "next/image";
import { useState, useEffect } from "react";
// import Markdown from "marked-react";
// import Markdown from 'markdown-to-jsx'
// import { render } from 'react-dom'
import ReactMarkdown from 'react-markdown'
import { marked } from 'marked';
import { v4 as uuidv4 } from 'uuid';
// import { ChevronLeft, ChevronRight } from 'lucide-react'
import { FaSearch } from 'react-icons/fa';
import { FaChevronLeft, FaChevronRight } from "react-icons/fa";
import { useRouter } from 'next/navigation'
import { useRef } from 'react';

export default function Home() {
  const router = useRouter()
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [inputColor, setInputColor] = useState('black')
  const [loading, setLoading] = useState(false);
  const [sessionID, setSessionID] = useState(null);
  const [showSidebar, setShowSidebar] = useState(true)
  const [collapsed, setCollapsed] = useState(false);
  const bottomRef = useRef(null);

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
  ]

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  useEffect(() => {
    // If the messages array is empty, create a sessionID
    if (messages.length === 0 && !sessionID) {
      const generatedSessionID = uuidv4(); // Generate a random UUID
      setSessionID(generatedSessionID);
    }
  }, [messages, sessionID]);

  const handleInputChange = (e) => {
    setInput(e.target.value);
    setInputColor('black');
  };

  const sendMessage = async () => {
    if (!input.trim()) return; // If input is empty, do nothing.
    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]); // Add user message to chat
    setInput(""); // Clear the input
    setLoading(true); // Start loading indicator

    try {
      const response = await fetch("http://192.168.30.172:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: input, user: "testuser", sessionID: sessionID }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch response from the server');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      let botMessage = { role: "assistant", content: "" }; // Initial empty AI message
      setMessages((prev) => [...prev, botMessage]); // Add empty message to array

      const readStream = async () => {
        const { value, done } = await reader.read();
        if (done) {
          setLoading(false); // Stop loading once the stream is finished
          return;
        }

        let token = decoder.decode(value, { stream: true }); // Decode the chunk of data

        setMessages((prev) => {
          let updatedMessages = [...prev];
          updatedMessages[updatedMessages.length - 1] = {
            role: "assistant",
            content: updatedMessages[updatedMessages.length - 1].content + token, // Append token
          };
          return updatedMessages;
        });

        // Recursively call readStream to continue reading
        if (!done) {
          readStream();
        }
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



  return (
    <div className="flex flex-col h-screen">
      <header style={{ backgroundColor: "#09195d", position: "fixed", top: 0, width: "100%", height: "80px" }} className="text-white py-4 px-8 shadow-md relative flex items-center">
        {/* Logo */}
        <div className="flex items-center gap-4">
          <Image src="/veron-logo-rm.png" alt="Veron Logo" width={160} height={40} />
        </div>

        {/* Centered Title */}
        <div className="absolute left-1/2 transform -translate-x-1/2 text-3xl font-bold tracking-wide flex items-center gap-4">
          <h1>VERON Chatbot</h1> {/* Added margin-right to separate text from image */}
          <Image src="/robo.png" alt="Veron Logo" width={80} height={10} />
        </div>

        {/* Right Label */}
        <span className="ml-auto text-gray-300 text-sm">Powered by VERON R&D</span>
      </header>

      <main className="flex-1 w-full flex flex-col gap-8 row-start-2 pt-20 min-h-0">


        <div className="w-full flex flex-col flex-1 min-h-0">
          <div className="flex flex-row justify-between bg-white h-full">
            <div className={`flex flex-col ${collapsed ? 'w-16' : 'w-1/6'} border-r-2 transition-all duration-300 overflow-y-auto `}>
              <div className="flex justify-between border-b-2  py-4 px-2">

                {!collapsed && (

                  <div>

                    <div className="">
                      <div className="flex items-center border-2 border-gray-200 rounded-2xl px-2">
                        <input
                          type="text"
                          placeholder="Search chatting"
                          className="py-2 px-2 w-full focus:outline-none rounded-2xl"
                        />
                        <FaSearch className="text-gray-500 h-4 w-4 ml-2" />
                      </div>
                    </div>

                  </div>
                )}
                {/* Collapse/Expand Button */}
                <div className="flex justify-end p-2">
                  <button onClick={() => setCollapsed(!collapsed)}>
                    {collapsed ? (
                      <FaChevronRight className="text-gray-600" />
                    ) : (
                      <FaChevronLeft className="text-gray-600" />
                    )}
                  </button>
                </div>

              </div>


              {/* Search Input */}
              {!collapsed && (
                <div>


                  {chatList.map((chat, idx) => (
                    <ChatItem key={idx} {...chat} />
                  ))}
                </div>
              )}

            </div>

            <div
              className="w-full flex flex-col justify-between bg-cover bg-center overflow-hidden flex-1 min-h-0"
              style={{
                backgroundImage: "url('/background.jpg')"
              }}
            >
              <div className="flex flex-col flex-1 overflow-y-auto min-h-0 scrollbar-custom pb-12">

                {/* <div className="border p-4 h-80 overflow-y-auto"> */}
                {messages.map((msg, index) => (
                  <div key={index}>
                    {msg.role == "user" ? <div className="flex justify-end mb-2 mt-3 mr-3">
                      <div
                        className="mr-2 py-3 px-4 bg-blue-400 rounded-bl-3xl rounded-tl-3xl rounded-tr-xl text-white"
                      >
                        {msg.content}
                      </div>
                      <img
                        src="/avt.jpg"
                        className="object-cover h-12 w-12 rounded-full"
                        alt=""
                      />
                    </div> : <div className="flex ml-3 mb-4 w-full sm:w-2/3">
                      <img
                        src="/robo-avt.png"
                        className="object-cover h-12 w-12 rounded-full"
                        alt=""
                      />
                      <div
                        className="ml-2 py-3 px-4 bg-gray-200 rounded-br-3xl rounded-tr-3xl rounded-tl-xl text-black"
                      >
                        {/* {msg.content} */}
                        <div dangerouslySetInnerHTML={{ __html: marked(msg.content) }} />
                      </div>
                    </div>}
                  </div>
                ))}
                <div ref={bottomRef} className="h-6" />
              </div>
              <div className="flex items-center gap-3 mb-4 mr-4 ml-4">


                <input
                  type="text"
                  value={input}
                  onChange={handleInputChange}
                  onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                  className="flex-1 border border-gray-300 rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Type your message..."
                  style={{ color: inputColor }}
                />
                <button
                  onClick={sendMessage}
                  style={{ backgroundColor: "#09195d" }}
                  className="hover:bg-blue-700 text-white px-5 py-2 rounded-full transition"
                >
                  Send
                </button>



              </div>
            </div>


          </div>
        </div>

      </main>
    </div>
  );
}
