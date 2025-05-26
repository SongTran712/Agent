"use client";
import Image from "next/image";
import { useState, useEffect } from "react";
// import Markdown from "marked-react";
// import Markdown from 'markdown-to-jsx'
// import { render } from 'react-dom'
import ReactMarkdown from 'react-markdown'
import { marked } from 'marked';
import { v4 as uuidv4 } from 'uuid';

export default function Home() {

  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sessionID, setSessionID] = useState(null);

  useEffect(() => {
    // If the messages array is empty, create a sessionID
    if (messages.length === 0 && !sessionID) {
      const generatedSessionID = uuidv4(); // Generate a random UUID
      setSessionID(generatedSessionID);
    }
  }, [messages, sessionID]);

  const sendMessage = async () => {
    if (!input.trim()) return; // If input is empty, do nothing.
    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]); // Add user message to chat
    setInput(""); // Clear the input
    setLoading(true); // Start loading indicator


  
    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: input, user:"testuser", sessionID: sessionID }),
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
    <div className="flex flex-col min-h-screen">
      <header className="bg-gray-800 text-white text-center py-8">
      <div className="inline-flex items-center">
  <h1 className="text-4xl font-bold pr-5">FUNNY DEMO  </h1>
  <Image
    aria-hidden
    src="/face.png"
    alt="File icon"
    width={50}
    height={50}
  />
</div>
        

        {/* <p className="text-lg mt-2">People detection</p> */}
      </header>

      <main className="flex flex-col gap-8 row-start-2 items-center sm:items-start">
       
      
<div className="container mx-auto shadow-lg rounded-lg">
      
    <div className="px-5 py-5 flex justify-between items-center bg-white border-b-2">
      <div className="font-semibold text-2xl">GoingChat</div>
      <div className="w-1/2">
        <p>Hello User, Welcome to Chatbox</p>
      </div>
      <div
        className="h-12 w-12 p-2 bg-yellow-500 rounded-full text-white font-semibold flex items-center justify-center"
      >
        RA
      </div>
    </div>
    <div className="flex flex-row justify-between bg-white">
      <div className="flex flex-col w-2/5 border-r-2 overflow-y-auto">
        
        <div className="border-b-2 py-4 px-2">
          <input
            type="text"
            placeholder="search chatting"
            className="py-2 px-2 border-2 border-gray-200 rounded-2xl w-full"
          />
        </div>
    
        <div
          className="flex flex-row py-4 px-2 justify-center items-center border-b-2"
        >
          <div className="w-1/4">
            <img
              src="https://source.unsplash.com/_7LbC5J-jw4/600x600"
              className="object-cover h-12 w-12 rounded-full"
              alt=""
            />
          </div>
          <div className="w-full">
            <div className="text-lg font-semibold">Luis1994</div>
            <span className="text-gray-500">Pick me at 9:00 Am</span>
          </div>
        </div>
        <div className="flex flex-row py-4 px-2 items-center border-b-2">
          <div className="w-1/4">
            <img
              src="https://source.unsplash.com/otT2199XwI8/600x600"
              className="object-cover h-12 w-12 rounded-full"
              alt=""
            />
          </div>
          <div className="w-full">
            <div className="text-lg font-semibold">Everest Trip 2021</div>
            <span className="text-gray-500">Hi Sam, Welcome</span>
          </div>
        </div>
        <div
          className="flex flex-row py-4 px-2 items-center border-b-2 border-l-4 border-blue-400"
        >
          <div className="w-1/4">
            <img
              src="https://source.unsplash.com/L2cxSuKWbpo/600x600"
              className="object-cover h-12 w-12 rounded-full"
              alt=""
            />
          </div>
          <div className="w-full">
            <div className="text-lg font-semibold">MERN Stack</div>
            <span className="text-gray-500">Lusi : Thanks Everyone</span>
          </div>
        </div>
        <div className="flex flex-row py-4 px-2 items-center border-b-2">
          <div className="w-1/4">
            <img
              src="https://source.unsplash.com/vpOeXr5wmR4/600x600"
              className="object-cover h-12 w-12 rounded-full"
              alt=""
            />
          </div>
          <div className="w-full">
            <div className="text-lg font-semibold">Javascript Indonesia</div>
            <span className="text-gray-500">Evan : some one can fix this</span>
          </div>
        </div>
        <div className="flex flex-row py-4 px-2 items-center border-b-2">
          <div className="w-1/4">
            <img
              src="https://source.unsplash.com/vpOeXr5wmR4/600x600"
              className="object-cover h-12 w-12 rounded-full"
              alt=""
            />
          </div>
          <div className="w-full">
            <div className="text-lg font-semibold">Javascript Indonesia</div>
            <span className="text-gray-500">Evan : some one can fix this</span>
          </div>
        </div>

        <div className="flex flex-row py-4 px-2 items-center border-b-2">
          <div className="w-1/4">
            <img
              src="https://source.unsplash.com/vpOeXr5wmR4/600x600"
              className="object-cover h-12 w-12 rounded-full"
              alt=""
            />
          </div>
          <div className="w-full">
            <div className="text-lg font-semibold">Javascript Indonesia</div>
            <span className="text-gray-500">Evan : some one can fix this</span>
          </div>
        </div>
      
      </div>
      
      <div className="w-full px-5 flex flex-col justify-between">
        <div className="flex flex-col mt-5">
        


        {/* <div className="border p-4 h-80 overflow-y-auto"> */}
        {messages.map((msg, index) => (
          <div key={index}> 
            {msg.role == "user" ? <div className="flex justify-end mb-4">
            <div
              className="mr-2 py-3 px-4 bg-blue-400 rounded-bl-3xl rounded-tl-3xl rounded-tr-xl text-white"
            >
              {msg.content}
            </div>
            <img
              src="https://source.unsplash.com/vpOeXr5wmR4/600x600"
              className="object-cover h-8 w-8 rounded-full"
              alt=""
            />
          </div>:<div className="flex justify-start mb-4">
            <img
              src="https://source.unsplash.com/vpOeXr5wmR4/600x600"
              className="object-cover h-8 w-8 rounded-full"
              alt=""
            />
            <div
              className="ml-2 py-3 px-4 bg-gray-400 rounded-br-3xl rounded-tr-3xl rounded-tl-xl text-white"
              >
              {/* {msg.content} */}
              <div dangerouslySetInnerHTML={{ __html: marked(msg.content) }} />
            </div>
          </div>}
          </div>
        ))}
      {/* </div> */}

          {/* <div className="flex justify-start mb-4">
            <img
              src="https://source.unsplash.com/vpOeXr5wmR4/600x600"
              className="object-cover h-8 w-8 rounded-full"
              alt=""
            />
            <div
              className="ml-2 py-3 px-4 bg-gray-400 rounded-br-3xl rounded-tr-3xl rounded-tl-xl text-white"
              >
              Lorem ipsum dolor sit amet consectetur adipisicing elit. Quaerat
              at praesentium, aut ullam delectus odio error sit rem. Architecto
              nulla doloribus laborum illo rem enim dolor odio saepe,
              consequatur quas?
            </div>
          </div> */}
          
         
        </div>
        <div className="py-5 flex items-center gap-3">
  <input
    className="w-full bg-gray-300 py-3 px-3 rounded-xl"
    type="text"
    placeholder="Type your message here..."
    onChange={(event)=>{setInput(event.target.value)}}
    value = {input}
  />
  <button className="bg-blue-500 text-white px-5 py-3 rounded-xl"
  onClick={()=>{sendMessage()}}
  >
    Enter
  </button>
</div>
      </div>
      
      
      </div>
</div>

      </main>
      <footer className="row-start-3 flex gap-6 flex-wrap items-center justify-center">
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://nextjs.org/learn?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/file.svg"
            alt="File icon"
            width={16}
            height={16}
          />
          Learn
        </a>
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://vercel.com/templates?framework=next.js&utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/window.svg"
            alt="Window icon"
            width={16}
            height={16}
          />
          Examples
        </a>
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://nextjs.org?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/globe.svg"
            alt="Globe icon"
            width={16}
            height={16}
          />
          Go to nextjs.org →
        </a>
      </footer>
    </div>
  );
}
