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
    { imgSrc: '/chatbot.webp', name: 'Chatbot', lastMsg: 'Last message preview', route: '/chat' },
    { imgSrc: '/agent.avif', name: 'Agent', lastMsg: 'Last message preview', route: '/agent' },
    { imgSrc: '/vlm.jpg', name: 'VLMs', lastMsg: 'Last message preview', route: '/vlm' },
  ]


  const handleInputChange = (e) => {
    setInput(e.target.value);
    setInputColor('black');
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
               
 

                <iframe src="https://seltecio-my.sharepoint.com/personal/song_tran_veronlabs_com/_layouts/15/Doc.aspx?sourcedoc={6092a95e-066a-49f3-864e-b200ed30a2bf}&amp;action=embedview&amp;wdAr=1.7777777777777777" 
                width="100%" height="100%" 
                frameborder="0">This is an embedded <a target="_blank" href="https://office.com">Microsoft Office</a> presentation, powered by <a target="_blank" href="https://office.com/webapps">Office</a>.</iframe>


              </div>
            </div>


          </div>
        </div>

      </main>
    </div>
  );
}
