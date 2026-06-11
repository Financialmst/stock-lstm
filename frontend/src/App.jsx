import { useState } from "react";

import axios from "axios";


function App() {

  const [ticker, setTicker] = useState("");

  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);


  const analyzeStock = async () => {

    if (!ticker) return;

    setLoading(true);

    try {

      const response = await axios.get(

        `http://127.0.0.1:5000/analyze/${ticker}`

      );

      setResult(response.data);

    } catch (error) {

      console.error(error);

      alert("Backend not running");

    }

    setLoading(false);
  };


  return (

    <div
      style={{
        minHeight: "100vh",
        background: "#020617",
        color: "white",
        padding: "40px",
        fontFamily: "Arial"
      }}
    >

      {/* -------------------- */}
      {/* TITLE */}
      {/* -------------------- */}
      <h1
        style={{
          fontSize: "52px",
          textAlign: "center",
          marginBottom: "40px",
          fontWeight: "bold"
        }}
      >
        AI Stock Intelligence Platform
      </h1>


      {/* -------------------- */}
      {/* SEARCH BAR */}
      {/* -------------------- */}
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          gap: "15px"
        }}
      >

        <input

          type="text"

          placeholder="Enter ticker (AAPL)"

          value={ticker}

          onChange={(e) =>
            setTicker(e.target.value)
          }

          style={{
            padding: "16px",
            width: "350px",
            fontSize: "18px",
            borderRadius: "14px",
            border: "none",
            outline: "none"
          }}
        />

        <button

          onClick={analyzeStock}

          style={{
            padding: "16px 28px",
            fontSize: "18px",
            borderRadius: "14px",
            border: "none",
            cursor: "pointer",
            background: "#22c55e",
            color: "white",
            fontWeight: "bold"
          }}
        >
          Analyze
        </button>

      </div>


      {/* -------------------- */}
      {/* LOADING */}
      {/* -------------------- */}
      {loading && (

        <h2
          style={{
            marginTop: "50px",
            textAlign: "center"
          }}
        >
          AI Agents Analyzing Market...
        </h2>

      )}


      {/* -------------------- */}
      {/* RESULT CARD */}
      {/* -------------------- */}
      {result && (

        <div
          style={{
            marginTop: "50px",
            background: "#111827",
            padding: "40px",
            borderRadius: "24px",
            maxWidth: "800px",
            marginInline: "auto",
            boxShadow:
              "0px 0px 25px rgba(0,0,0,0.4)"
          }}
        >

          {/* Ticker */}
          <h2
            style={{
              fontSize: "42px",
              marginBottom: "25px"
            }}
          >
            {result.ticker}
          </h2>


          {/* GRID */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                "1fr 1fr",
              gap: "20px"
            }}
          >

            {/* Current Price */}
            <div
              style={cardStyle}
            >
              <h3>Current Price</h3>

              <p style={valueStyle}>
                ${result.current_price}
              </p>
            </div>


            {/* Predicted Price */}
            <div
              style={cardStyle}
            >
              <h3>Predicted Price</h3>

              <p style={valueStyle}>
                ${result.predicted_price}
              </p>
            </div>


            {/* Trend */}
            <div
              style={cardStyle}
            >
              <h3>Trend</h3>

              <p style={valueStyle}>
                {result.trend}
              </p>
            </div>


            {/* Sentiment */}
            <div
              style={cardStyle}
            >
              <h3>Sentiment</h3>

              <p style={valueStyle}>
                {result.sentiment}
              </p>
            </div>


            {/* Decision */}
            <div
              style={cardStyle}
            >
              <h3>Decision</h3>

              <p
                style={{
                  ...valueStyle,

                  color:

                    result.decision === "BUY"
                    ? "#22c55e"

                    : result.decision === "SELL"
                    ? "#ef4444"

                    : "#facc15"
                }}
              >
                {result.decision}
              </p>
            </div>


            {/* Confidence */}
            <div
              style={cardStyle}
            >
              <h3>Confidence</h3>

              <p style={valueStyle}>
                {result.confidence}%
              </p>
            </div>


            {/* Risk */}
            <div
              style={cardStyle}
            >
              <h3>Risk Level</h3>

              <p style={valueStyle}>
                {result.risk_level}
              </p>
            </div>


            {/* Volatility */}
            <div
              style={cardStyle}
            >
              <h3>Volatility</h3>

              <p style={valueStyle}>
                {result.volatility_percent}%
              </p>
            </div>

          </div>
          {/* RSI */}
<div style={cardStyle}>

  <h3>RSI</h3>

  <p style={valueStyle}>
    {result.rsi}
  </p>

</div>


{/* RSI Signal */}
<div style={cardStyle}>

  <h3>RSI Signal</h3>

  <p style={valueStyle}>
    {result.rsi_signal}
  </p>

</div>


{/* MACD */}
<div style={cardStyle}>

  <h3>MACD</h3>

  <p style={valueStyle}>
    {result.macd}
  </p>

</div>


{/* MACD Signal */}
<div style={cardStyle}>

  <h3>MACD Signal</h3>

  <p style={valueStyle}>
    {result.macd_signal}
  </p>

</div>


{/* SMA 20 */}
<div style={cardStyle}>

  <h3>SMA 20</h3>

  <p style={valueStyle}>
    {result.sma_20}
  </p>

</div>


{/* SMA 50 */}
<div style={cardStyle}>

  <h3>SMA 50</h3>

  <p style={valueStyle}>
    {result.sma_50}
  </p>

</div>


{/* Technical Trend */}
<div style={cardStyle}>

  <h3>Technical Trend</h3>

  <p style={valueStyle}>
    {result.technical_trend}
  </p>

</div>
{/* MARKET REGIME */}
<div style={cardStyle}>

  <h3>Market Regime</h3>

  <p style={valueStyle}>
    {result.market_regime}
  </p>

</div>


{/* MARKET DIRECTION */}
<div style={cardStyle}>

  <h3>Market Direction</h3>

  <p style={valueStyle}>
    {result.market_direction}
  </p>

</div>


{/* BULLISH PROBABILITY */}
<div style={cardStyle}>

  <h3>Bullish Probability</h3>

  <p style={valueStyle}>
    {result.bullish_probability}%
  </p>

</div>


{/* BEARISH PROBABILITY */}
<div style={cardStyle}>

  <h3>Bearish Probability</h3>

  <p style={valueStyle}>
    {result.bearish_probability}%
  </p>

</div>


{/* OVERALL CONFIDENCE */}
<div style={cardStyle}>

  <h3>Overall Confidence</h3>

  <p style={valueStyle}>
    {result.overall_confidence}%
  </p>

</div>


{/* AI EXPLANATION */}
<div
  style={{
    marginTop: "35px",
    background: "#1e293b",
    padding: "25px",
    borderRadius: "18px"
  }}
>

  <h3
    style={{
      marginBottom: "15px"
    }}
  >
    AI Explanation
  </h3>

  <p
    style={{
      fontSize: "18px",
      lineHeight: "1.7"
    }}
  >
    {result.llm_reasoning}
  </p>

</div>


{/* MULTI-TIMEFRAME ANALYSIS */}
<div
  style={{
    marginTop: "35px",
    background: "#1e293b",
    padding: "25px",
    borderRadius: "18px"
  }}
>

  <h3
    style={{
      marginBottom: "20px"
    }}
  >
    Multi-Timeframe Analysis
  </h3>

  {

    Object.entries(
      result.timeframes
    ).map(

      ([tf, data]) => (

        <div
          key={tf}
          style={{
            marginBottom: "15px",
            fontSize: "18px"
          }}
        >

          <strong>{tf}</strong>

          {" → "}

          {data.trend}

          {" ("}

          {data.percent_change}%

          {")"}

        </div>

      )
    )
  }

</div>

</div>

)}

</div>
);
}


/* -------------------- */
/* CARD STYLE */
/* -------------------- */
const cardStyle = {

  background: "#1e293b",

  padding: "20px",

  borderRadius: "18px"
};


/* -------------------- */
/* VALUE STYLE */
/* -------------------- */
const valueStyle = {

  fontSize: "28px",

  fontWeight: "bold",

  marginTop: "10px"
};


export default App;
