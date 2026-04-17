import { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Cell, CartesianGrid,
} from "recharts";

// ── All The Paws Brand Tokens ─────────────────────────────────────────────────
const B = {
  black:   "#1A1A1A",
  ice:     "#B8D0D4",
  iceL:    "#E8F2F4",
  iceDark: "#8AACB0",
  red:     "#E8451A",
  redL:    "#FDE8E3",
  amber:   "#F5A623",
  amberL:  "#FEF3E2",
  white:   "#FFFFFF",
  offWhite:"#F7F7F7",
  gray:    "#888888",
  grayL:   "#EEEEEE",
};

const API_BASE = "https://pawsiq-api-production.up.railway.app";
const api = {
  demandHeatmap:   (m,y) => fetch(`${API_BASE}/predict/demand/heatmap?month=${m}&year=${y}`).then(r=>r.json()),
  pricingSchedule: (m)   => fetch(`${API_BASE}/predict/price/schedule?month=${m}`).then(r=>r.json()),
  bookingSummary:  ()    => fetch(`${API_BASE}/bookings/summary`).then(r=>r.json()),
  walkers:         ()    => fetch(`${API_BASE}/walkers`).then(r=>r.json()),
  predictDemand:   (b)   => fetch(`${API_BASE}/predict/demand`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)}).then(r=>r.json()),
  predictPrice:    (b)   => fetch(`${API_BASE}/predict/price`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)}).then(r=>r.json()),
};

const MOCK = {
  summary:{total_bookings:9935,completed_bookings:9279,completion_rate_pct:93.4,
    total_revenue:240963.41,avg_surge:1.1873,peak_hour_pct:58.4,
    peak_avg_price:27.94,offpeak_avg_price:23.21,revenue_lift_vs_flat_pct:18.7,
    monthly_revenue:{"2024-07":9823,"2024-08":10241,"2024-09":11043,"2024-10":10789,"2024-11":9654,"2024-12":9112}},
  walkers:[
    {user_id:"W0001",name:"Amara Stevens",rating:4.4,total_walks:908},
    {user_id:"W0002",name:"Devon Park",rating:4.6,total_walks:1287},
    {user_id:"W0003",name:"Cleo Mitchell",rating:4.6,total_walks:1311},
    {user_id:"W0004",name:"Jordan Rivera",rating:4.8,total_walks:882},
    {user_id:"W0005",name:"Priya Okafor",rating:4.8,total_walks:1339},
    {user_id:"W0006",name:"Chris Laurent",rating:4.4,total_walks:1315},
    {user_id:"W0007",name:"Natalie Diaz",rating:4.8,total_walks:1274},
    {user_id:"W0008",name:"Marcus Chen",rating:4.9,total_walks:963},
  ],
  heatmap:(()=>{const rows=[],hw={7:1.8,8:1.9,9:1.7,10:0.9,11:0.9,12:0.8,13:0.8,14:0.7,15:0.7,16:0.9,17:1.5,18:1.6,19:1.4,20:0.6},dw=[1.2,1.2,1.2,1.2,1.2,0.8,0.8];for(let d=0;d<7;d++)for(let h=6;h<=20;h++)rows.push({day_of_week:d,hour_of_day:h,predicted:+(dw[d]*(hw[h]||0.6)*0.9).toFixed(2),is_peak:[7,8,9,17,18,19].includes(h)});return rows;})(),
  pricing:(()=>{const rows=[];for(let d=0;d<7;d++)for(let h=6;h<=20;h++){const s=Math.min(1.35,0.9+([7,8,9,17,18,19].includes(h)?0.3:0.1)+(d<5?0.05:0)+Math.random()*0.04);rows.push({day_of_week:d,hour_of_day:h,service_type:"walk_30",surge:+s.toFixed(4),final_price:+(16*s).toFixed(2)});}return rows;})(),
};

const HEADING = "'Fredoka One', cursive";
const BODY    = "'Nunito', sans-serif";

const SectionLabel = ({children}) => (
  <p style={{fontFamily:HEADING,fontSize:12,letterSpacing:"0.12em",
    textTransform:"uppercase",color:B.gray,marginBottom:10}}>{children}</p>
);

const Card = ({children,style={}}) => (
  <div style={{background:B.white,border:`2.5px solid ${B.black}`,borderRadius:20,
    padding:"20px 22px",boxShadow:`4px 4px 0px ${B.black}`,...style}}>{children}</div>
);

const KpiCard = ({label,value,sub,color=B.black,bg=B.white}) => (
  <div style={{background:bg,border:`2.5px solid ${B.black}`,borderRadius:20,
    padding:"18px 20px",boxShadow:`4px 4px 0px ${B.black}`,flex:1,minWidth:140}}>
    <p style={{fontFamily:HEADING,fontSize:11,letterSpacing:"0.1em",
      textTransform:"uppercase",color:B.gray,marginBottom:6}}>{label}</p>
    <p style={{fontFamily:HEADING,fontSize:28,color,lineHeight:1,marginBottom:4}}>{value}</p>
    {sub&&<p style={{fontFamily:BODY,fontSize:11,color:B.gray}}>{sub}</p>}
  </div>
);

const Spinner = () => (
  <div style={{display:"flex",alignItems:"center",justifyContent:"center",
    height:120,fontFamily:HEADING,color:B.gray,fontSize:14,letterSpacing:"0.08em"}}>
     LOADING...
  </div>
);

const PawDivider = () => (
  <div style={{display:"flex",alignItems:"center",gap:6,margin:"2px 0"}}>
    <span style={{fontSize:9,opacity:0.25}}></span>
    <div style={{flex:1,height:1,background:B.grayL}}/>
  </div>
);

// ── Heatmap ───────────────────────────────────────────────────────────────────
const DOW   = ["MON","TUE","WED","THU","FRI","SAT","SUN"];
const HOURS = Array.from({length:15},(_,i)=>i+6);

function DemandHeatmap({data,loading}) {
  const [sel,setSel]=useState(null);
  if(loading) return <Spinner/>;
  const grid={};
  data.forEach(d=>{if(!grid[d.day_of_week])grid[d.day_of_week]={};grid[d.day_of_week][d.hour_of_day]=d.predicted;});
  const max=Math.max(...data.map(d=>d.predicted));
  const cellColor=v=>{const t=v/max;return t>0.75?B.red:t>0.5?B.amber:t>0.25?B.ice:B.iceL;};
  return(
    <div>
      <div style={{overflowX:"auto"}}>
        <table style={{borderCollapse:"separate",borderSpacing:"3px",minWidth:540}}>
          <thead><tr>
            <th style={{width:36,fontFamily:HEADING,fontSize:9,color:B.gray,
              fontWeight:400,paddingBottom:4}}/>
            {HOURS.map(h=>(
              <th key={h} style={{fontFamily:HEADING,fontSize:9,color:B.gray,
                fontWeight:400,paddingBottom:4,textAlign:"center",width:34}}>{h}</th>
            ))}
          </tr></thead>
          <tbody>{Array.from({length:7},(_,d)=>(
            <tr key={d}>
              <td style={{fontFamily:HEADING,fontSize:9,color:B.black,paddingRight:6,paddingBottom:2}}>{DOW[d]}</td>
              {HOURS.map(h=>{
                const val=grid[d]?.[h]??0,isSel=sel?.d===d&&sel?.h===h;
                return(
                  <td key={h} onMouseEnter={()=>setSel({d,h,val})}
                    onMouseLeave={()=>setSel(null)} style={{padding:"2px"}}>
                    <div style={{width:30,height:20,borderRadius:6,background:cellColor(val),
                      border:isSel?`2px solid ${B.black}`:`2px solid transparent`,
                      cursor:"pointer",transition:"all 0.1s",
                      display:"flex",alignItems:"center",justifyContent:"center"}}>
                      {isSel&&<span style={{fontFamily:HEADING,fontSize:7,color:B.black}}>
                        {val.toFixed(1)}</span>}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}</tbody>
        </table>
      </div>
      <div style={{display:"flex",gap:10,marginTop:12,alignItems:"center",flexWrap:"wrap"}}>
        {[[B.iceL,"LOW"],[B.ice,""],[B.amber,""],[B.red,"HIGH"]].map(([c,l],i)=>(
          <div key={i} style={{display:"flex",alignItems:"center",gap:4}}>
            <div style={{width:16,height:16,borderRadius:4,background:c,
              border:`1.5px solid ${B.black}33`}}/>
            {l&&<span style={{fontFamily:HEADING,fontSize:9,color:B.gray}}>{l}</span>}
          </div>
        ))}
        {sel&&(
          <div style={{marginLeft:8,background:B.amber,border:`2px solid ${B.black}`,
            borderRadius:10,padding:"3px 10px",fontFamily:HEADING,fontSize:10,color:B.black,
            boxShadow:`2px 2px 0 ${B.black}`}}>
            {DOW[sel.d]} {sel.h}:00 → {sel.val.toFixed(2)} bookings
          </div>
        )}
      </div>
    </div>
  );
}

// ── Pricing ───────────────────────────────────────────────────────────────────
function PricingPanel({data,loading}) {
  if(loading) return <Spinner/>;
  const monWalk=data.filter(d=>d.day_of_week===0&&d.service_type==="walk_30")
    .sort((a,b)=>a.hour_of_day-b.hour_of_day)
    .map(d=>({hour:`${d.hour_of_day}:00`,surge:d.surge,price:d.final_price,
      fill:[7,8,9,17,18,19].includes(d.hour_of_day)?B.red:d.surge>=1.10?B.amber:B.ice}));
  return(
    <div>
      <p style={{fontFamily:BODY,fontSize:12,color:B.gray,marginBottom:14}}>
        30-min walk · Monday · Ridge Regression model</p>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={monWalk} margin={{top:4,right:4,bottom:0,left:-10}}>
          <CartesianGrid strokeDasharray="3 3" stroke={B.grayL} vertical={false}/>
          <XAxis dataKey="hour" tick={{fontFamily:HEADING,fontSize:9,fill:B.gray}}
            tickLine={false} axisLine={false}/>
          <YAxis tick={{fontFamily:HEADING,fontSize:9,fill:B.gray}} tickLine={false}
            axisLine={false} domain={[0.8,1.4]} tickFormatter={v=>`×${v.toFixed(1)}`}/>
          <Tooltip content={({active,payload})=>active&&payload?.length?(
            <div style={{background:B.white,border:`2px solid ${B.black}`,borderRadius:12,
              padding:"10px 14px",boxShadow:`3px 3px 0 ${B.black}`}}>
              <p style={{fontFamily:HEADING,fontSize:11,color:B.black}}>{payload[0].payload.hour}</p>
              <p style={{fontFamily:HEADING,fontSize:13,color:B.amber}}>×{payload[0].value.toFixed(4)}</p>
              <p style={{fontFamily:BODY,fontSize:11,color:B.gray}}>${payload[0].payload.price}</p>
            </div>):null}/>
          <Bar dataKey="surge" radius={[6,6,0,0]} maxBarSize={26}>
            {monWalk.map((d,i)=><Cell key={i} fill={d.fill}/>)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div style={{display:"flex",gap:8,marginTop:10,flexWrap:"wrap"}}>
        {[["🔴 PEAK ×1.25+",B.red,B.redL],["🟠 MEDIUM ×1.10+",B.amber,B.amberL],
          ["🔵 STANDARD",B.iceDark,B.iceL]].map(([l,c,bg])=>(
          <div key={l} style={{background:bg,border:`2px solid ${c}`,borderRadius:100,
            padding:"3px 10px",fontFamily:HEADING,fontSize:9,color:c}}>{l}</div>
        ))}
      </div>
    </div>
  );
}

// ── Walkers ───────────────────────────────────────────────────────────────────
function WalkerTable({walkers,loading}) {
  if(loading) return <Spinner/>;
  const sorted=[...walkers].sort((a,b)=>b.total_walks-a.total_walks);
  const maxW=Math.max(...sorted.map(w=>w.total_walks));
  return(
    <div style={{display:"flex",flexDirection:"column",gap:12}}>
      {sorted.map((w,i)=>(
        <div key={w.user_id}>
          <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:4}}>
            <div style={{width:32,height:32,borderRadius:"50%",
              background:i<3?B.amber:B.ice,border:`2px solid ${B.black}`,
              flexShrink:0,display:"flex",alignItems:"center",justifyContent:"center",
              fontFamily:HEADING,fontSize:12,color:B.black}}>{w.name[0]}</div>
            <div style={{flex:1}}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                <span style={{fontFamily:HEADING,fontSize:13,color:B.black}}>{w.name}</span>
                <span style={{fontFamily:HEADING,fontSize:11,color:B.gray}}>
                  ★{w.rating} · {w.total_walks.toLocaleString()}</span>
              </div>
              <div style={{height:6,borderRadius:100,background:B.grayL,
                marginTop:4,overflow:"hidden",border:`1.5px solid ${B.black}22`}}>
                <div style={{height:"100%",borderRadius:100,
                  background:w.rating>=4.8?B.amber:B.ice,
                  width:`${(w.total_walks/maxW)*100}%`,transition:"width 0.6s ease"}}/>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Live Predictor ────────────────────────────────────────────────────────────
function LivePredictor({apiOnline}) {
  const [hour,setHour]=useState(8);
  const [dow,setDow]=useState(0);
  const [svc,setSvc]=useState("walk_30");
  const [result,setResult]=useState(null);
  const [loading,setLoading]=useState(false);
  const BASE={walk_30:16,walk_60:24,drop_in:14,overnight:55};

  const predict=async()=>{
    setLoading(true);
    try{
      if(apiOnline){
        const [demand,price]=await Promise.all([
          api.predictDemand({hour_of_day:hour,day_of_week:dow,month:4,year:2025}),
          api.predictPrice({hour_of_day:hour,day_of_week:dow,month:4,service_type:svc,zip_hour_demand:2}),
        ]);
        setResult({demand,price});
      } else {
        const isPeak=[7,8,9,17,18,19].includes(hour);
        const surge=Math.min(1.35,0.9+(isPeak?0.3:0.1)+(dow<5?0.05:0));
        setResult({
          demand:{predicted_bookings:isPeak?2.1:0.6,demand_level:isPeak?"medium":"low",is_peak_hour:isPeak},
          price:{surge_multiplier:+surge.toFixed(4),base_price:BASE[svc],
            final_price:+(BASE[svc]*surge).toFixed(2),
            pricing_tier:surge>=1.25?"high":surge>=1.10?"medium":"standard",
            price_explanation:isPeak?"Peak hour — surge applied":"Standard pricing"},
        });
      }
    }catch(e){console.error(e);}
    setLoading(false);
  };

  const sel={background:B.offWhite,border:`2px solid ${B.black}`,borderRadius:10,
    padding:"8px 12px",fontSize:12,color:B.black,cursor:"pointer",fontFamily:BODY,
    fontWeight:600,outline:"none"};

  return(
    <div>
      <p style={{fontFamily:BODY,fontSize:12,color:B.gray,marginBottom:14}}>
        Pick a time slot to call both ML models live </p>
      <div style={{display:"flex",gap:8,flexWrap:"wrap",marginBottom:16}}>
        <select value={hour} onChange={e=>{setHour(+e.target.value);setResult(null);}} style={sel}>
          {HOURS.map(h=><option key={h} value={h}>{h}:00</option>)}
        </select>
        <select value={dow} onChange={e=>{setDow(+e.target.value);setResult(null);}} style={sel}>
          {["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].map((d,i)=><option key={i} value={i}>{d}</option>)}
        </select>
        <select value={svc} onChange={e=>{setSvc(e.target.value);setResult(null);}} style={sel}>
          <option value="walk_30">30-min Walk</option>
          <option value="walk_60">60-min Walk</option>
          <option value="drop_in">Drop-in</option>
          <option value="overnight">Overnight</option>
        </select>
        <button onClick={predict} disabled={loading} style={{
          background:loading?B.grayL:B.black,color:loading?B.gray:B.white,
          border:`2px solid ${B.black}`,borderRadius:10,padding:"8px 20px",
          fontFamily:HEADING,fontSize:13,cursor:loading?"not-allowed":"pointer",
          boxShadow:loading?"none":`3px 3px 0 ${B.amber}`,
          transition:"all 0.15s",letterSpacing:"0.05em"}}>
          {loading?"...":"PREDICT "}
        </button>
      </div>
      {result&&(
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
          <div style={{background:B.iceL,border:`2.5px solid ${B.iceDark}`,
            borderRadius:16,padding:"16px",boxShadow:`3px 3px 0 ${B.iceDark}`}}>
            <p style={{fontFamily:HEADING,fontSize:10,letterSpacing:"0.1em",
              color:B.iceDark,marginBottom:6}}>DEMAND FORECAST</p>
            <p style={{fontFamily:HEADING,fontSize:28,color:B.black,lineHeight:1}}>
              {result.demand.predicted_bookings.toFixed(2)}</p>
            <p style={{fontFamily:BODY,fontSize:11,color:B.gray,marginTop:4}}>
              bookings/slot · <strong>{result.demand.demand_level}</strong>
              {result.demand.is_peak_hour&&" · ⚡ peak"}</p>
          </div>
          <div style={{
            background:result.price.pricing_tier==="high"?B.redL:B.amberL,
            border:`2.5px solid ${result.price.pricing_tier==="high"?B.red:B.amber}`,
            borderRadius:16,padding:"16px",
            boxShadow:`3px 3px 0 ${result.price.pricing_tier==="high"?B.red:B.amber}`}}>
            <p style={{fontFamily:HEADING,fontSize:10,letterSpacing:"0.1em",
              color:result.price.pricing_tier==="high"?B.red:B.amber,marginBottom:6}}>DYNAMIC PRICE</p>
            <p style={{fontFamily:HEADING,fontSize:28,color:B.black,lineHeight:1}}>
              ${result.price.final_price}</p>
            <p style={{fontFamily:BODY,fontSize:11,color:B.gray,marginTop:4}}>
              ×{result.price.surge_multiplier} · {result.price.price_explanation}</p>
          </div>
        </div>
      )}
      {!apiOnline&&(
        <div style={{marginTop:10,background:B.amberL,border:`2px solid ${B.amber}`,
          borderRadius:10,padding:"6px 12px",display:"inline-block"}}>
          <span style={{fontFamily:HEADING,fontSize:9,color:B.amber,letterSpacing:"0.1em"}}>
            ⚠ API OFFLINE — MOCK PREDICTIONS
          </span>
        </div>
      )}
    </div>
  );
}

// ── Revenue Chart ─────────────────────────────────────────────────────────────
function RevenueChart({data}) {
  const chartData=Object.entries(data).map(([k,v])=>({month:k.slice(5),revenue:v}));
  return(
    <ResponsiveContainer width="100%" height={130}>
      <BarChart data={chartData} margin={{top:4,right:4,bottom:0,left:-10}}>
        <CartesianGrid strokeDasharray="3 3" stroke={B.grayL} vertical={false}/>
        <XAxis dataKey="month" tick={{fontFamily:HEADING,fontSize:9,fill:B.gray}}
          tickLine={false} axisLine={false}/>
        <YAxis tick={{fontFamily:HEADING,fontSize:9,fill:B.gray}} tickLine={false}
          axisLine={false} tickFormatter={v=>`$${(v/1000).toFixed(0)}k`}/>
        <Tooltip content={({active,payload})=>active&&payload?.length?(
          <div style={{background:B.white,border:`2px solid ${B.black}`,borderRadius:10,
            padding:"8px 12px",boxShadow:`2px 2px 0 ${B.black}`}}>
            <p style={{fontFamily:HEADING,fontSize:12,color:B.amber}}>
              ${payload[0].value.toLocaleString()}</p>
          </div>):null}/>
        <Bar dataKey="revenue" radius={[6,6,0,0]} maxBarSize={32}>
          {chartData.map((_,i)=><Cell key={i} fill={i===chartData.length-1?B.amber:B.ice}/>)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────
const TABS=[
  {id:"overview",label:" OVERVIEW"},
  {id:"demand",  label:" DEMAND"},
  {id:"pricing", label:" PRICING"},
  {id:"walkers", label:" WALKERS"},
  {id:"predict", label:" PREDICT"},
];

export default function Dashboard() {
  const [tab,setTab]         = useState("overview");
  const [summary,setSummary] = useState(null);
  const [walkers,setWalkers] = useState([]);
  const [heatmap,setHeatmap] = useState([]);
  const [pricing,setPricing] = useState([]);
  const [apiOnline,setApi]   = useState(false);
  const [loading,setLoading] = useState(true);
  const [ready,setReady]     = useState(false);

  useEffect(()=>{
    (async()=>{
      try{
        const [s,w,h,p]=await Promise.all([
          api.bookingSummary(),api.walkers(),
          api.demandHeatmap(4,2025),api.pricingSchedule(4)
        ]);
        setSummary(s);setWalkers(w);setHeatmap(h.heatmap);setPricing(p.schedule);setApi(true);
      }catch{
        setSummary(MOCK.summary);setWalkers(MOCK.walkers);
        setHeatmap(MOCK.heatmap);setPricing(MOCK.pricing);
      }
      setLoading(false);setTimeout(()=>setReady(true),50);
    })();
  },[]);

  return(
    <div style={{minHeight:"100vh",background:B.iceL,fontFamily:BODY}}>
      <link href="https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@400;600;700&display=swap" rel="stylesheet"/>

      {/* Header */}
      <header style={{background:B.black,padding:"16px 32px",
        display:"flex",alignItems:"center",justifyContent:"space-between",
        borderBottom:`3px solid ${B.amber}`}}>
        <div style={{display:"flex",alignItems:"center",gap:14}}>
          <div style={{width:42,height:42,borderRadius:"50%",background:B.amber,
            border:`3px solid ${B.white}`,display:"flex",alignItems:"center",
            justifyContent:"center",fontSize:20,boxShadow:`2px 2px 0 ${B.white}`}}></div>
          <div>
            <p style={{fontFamily:HEADING,fontSize:10,letterSpacing:"0.15em",
              color:B.amber,marginBottom:2}}>ALL THE PAWS</p>
            <h1 style={{fontFamily:HEADING,fontSize:20,color:B.white,margin:0,
              letterSpacing:"0.03em"}}>ML INTELLIGENCE DASHBOARD</h1>
          </div>
        </div>
        <div style={{display:"flex",alignItems:"center",gap:8,
          background:apiOnline?"#1A3A1A":"#3A1A1A",
          border:`2px solid ${apiOnline?"#4CAF50":B.red}`,
          borderRadius:100,padding:"6px 14px"}}>
          <div style={{width:7,height:7,borderRadius:"50%",
            background:apiOnline?"#4CAF50":B.red,
            boxShadow:`0 0 6px ${apiOnline?"#4CAF50":B.red}`}}/>
          <span style={{fontFamily:HEADING,fontSize:10,
            color:apiOnline?"#4CAF50":B.red,letterSpacing:"0.1em"}}>
            {apiOnline?"API LIVE":"MOCK DATA"}
          </span>
        </div>
      </header>

      {/* Tabs */}
      <nav style={{background:B.white,borderBottom:`2.5px solid ${B.black}`,
        padding:"0 32px",display:"flex",gap:0,overflowX:"auto"}}>
        {TABS.map(t=>(
          <button key={t.id} onClick={()=>setTab(t.id)} style={{
            fontFamily:HEADING,fontSize:11,letterSpacing:"0.08em",
            padding:"14px 18px",background:"none",border:"none",cursor:"pointer",
            color:tab===t.id?B.black:B.gray,whiteSpace:"nowrap",
            borderBottom:tab===t.id?`3px solid ${B.amber}`:"3px solid transparent",
            transition:"all 0.15s"}}>{t.label}
          </button>
        ))}
      </nav>

      {/* Content */}
      <main style={{padding:"28px 32px",maxWidth:1100,margin:"0 auto",
        opacity:ready?1:0,transition:"opacity 0.4s"}}>

        {/* OVERVIEW */}
        {tab==="overview"&&(
          <div style={{display:"flex",flexDirection:"column",gap:20}}>
            <div style={{display:"flex",gap:14,flexWrap:"wrap"}}>
              <KpiCard label="Total Revenue" color={B.amber} bg={B.amberL}
                value={summary?`$${(summary.total_revenue/1000).toFixed(0)}k`:"—"}
                sub="2-year synthetic dataset"/>
              <KpiCard label="Walks Completed"
                value={summary?.completed_bookings?.toLocaleString()??"—"}
                sub={`${summary?.completion_rate_pct??"—"}% completion`}/>
              <KpiCard label="Avg Surge" color={B.red} bg={B.redL}
                value={summary?`×${summary.avg_surge.toFixed(3)}`:"—"}
                sub={`+${summary?.revenue_lift_vs_flat_pct??"—"}% vs flat rate`}/>
              <KpiCard label="Peak Hour %" color={B.iceDark} bg={B.iceL}
                value={summary?`${summary.peak_hour_pct}%`:"—"}
                sub={`$${summary?.peak_avg_price??"—"} vs $${summary?.offpeak_avg_price??"—"}`}/>
            </div>
            <Card>
              <SectionLabel>Monthly Revenue — Last 6 Months</SectionLabel>
              {summary?<RevenueChart data={summary.monthly_revenue}/>:<Spinner/>}
            </Card>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
              <Card>
                <SectionLabel> Demand Model · GradientBoosting</SectionLabel>
                {[["MAE","0.738 bookings/slot"],["RMSE","0.912"],["R²","0.098"],["RMSE vs baseline","+5.0%"]].map(([k,v],i)=>(
                  <div key={k}>
                    <div style={{display:"flex",justifyContent:"space-between",padding:"8px 0",fontSize:13}}>
                      <span style={{color:B.gray,fontWeight:600}}>{k}</span>
                      <span style={{fontFamily:HEADING,fontSize:13,color:B.black}}>{v}</span>
                    </div>
                    {i<3&&<PawDivider/>}
                  </div>
                ))}
              </Card>
              <Card>
                <SectionLabel> Pricing Model · Ridge Regression</SectionLabel>
                {[["MAE","0.0569 ×surge"],["RMSE","0.0679"],["R²","0.727"],["Revenue lift","+17.1%"]].map(([k,v],i)=>(
                  <div key={k}>
                    <div style={{display:"flex",justifyContent:"space-between",padding:"8px 0",fontSize:13}}>
                      <span style={{color:B.gray,fontWeight:600}}>{k}</span>
                      <span style={{fontFamily:HEADING,fontSize:13,
                        color:k==="Revenue lift"?B.amber:B.black}}>{v}</span>
                    </div>
                    {i<3&&<PawDivider/>}
                  </div>
                ))}
              </Card>
            </div>
            <div style={{textAlign:"center",padding:"4px 0"}}>
              <span style={{fontFamily:HEADING,fontSize:11,color:B.gray,letterSpacing:"0.2em"}}>
                ALL THE PAWS · EST. 2025 · POWERED BY ML 
              </span>
            </div>
          </div>
        )}

        {/* DEMAND */}
        {tab==="demand"&&(
          <div style={{display:"flex",flexDirection:"column",gap:20}}>
            <Card>
              <SectionLabel> Demand Heatmap — April 2025</SectionLabel>
              <p style={{fontFamily:BODY,fontSize:12,color:B.gray,marginBottom:16}}>
                Predicted bookings per hour slot. Morning (7–9am) and evening (5–7pm) peaks
                learned from 2 years of NJ walk data. Hover a cell to inspect.</p>
              <DemandHeatmap data={heatmap} loading={loading}/>
            </Card>
            <Card>
              <SectionLabel> Avg Demand by Hour</SectionLabel>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={HOURS.map(h=>({hour:`${h}:00`,
                  avg:heatmap.filter(d=>d.hour_of_day===h).reduce((s,d)=>s+d.predicted,0)/
                    Math.max(1,heatmap.filter(d=>d.hour_of_day===h).length)
                }))} margin={{top:4,right:4,bottom:0,left:-10}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={B.grayL} vertical={false}/>
                  <XAxis dataKey="hour" tick={{fontFamily:HEADING,fontSize:9,fill:B.gray}}
                    tickLine={false} axisLine={false}/>
                  <YAxis tick={{fontFamily:HEADING,fontSize:9,fill:B.gray}}
                    tickLine={false} axisLine={false}/>
                  <Tooltip content={({active,payload})=>active&&payload?.length?(
                    <div style={{background:B.white,border:`2px solid ${B.black}`,
                      borderRadius:10,padding:"8px 12px",boxShadow:`2px 2px 0 ${B.black}`}}>
                      <p style={{fontFamily:HEADING,fontSize:12}}>{payload[0].value.toFixed(2)} avg</p>
                    </div>):null}/>
                  <Bar dataKey="avg" radius={[6,6,0,0]} maxBarSize={28}>
                    {HOURS.map((h,i)=><Cell key={i} fill={[7,8,9,17,18,19].includes(h)?B.red:B.amber}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        )}

        {/* PRICING */}
        {tab==="pricing"&&(
          <div style={{display:"flex",flexDirection:"column",gap:20}}>
            <Card>
              <SectionLabel> Surge by Hour · Ridge Regression · Monday</SectionLabel>
              <PricingPanel data={pricing} loading={loading}/>
            </Card>
            <Card>
              <SectionLabel>🐕 Pricing Tiers by Service</SectionLabel>
              <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:12,marginTop:8}}>
                {["walk_30","walk_60","drop_in","overnight"].map(svc=>{
                  const sd=pricing.filter(d=>d.service_type===svc);
                  const high=sd.filter(d=>d.surge>=1.25).length;
                  const med=sd.filter(d=>d.surge>=1.10&&d.surge<1.25).length;
                  const std=sd.filter(d=>d.surge<1.10).length;
                  const tot=sd.length||1;
                  const avg=sd.length?(sd.reduce((s,d)=>s+d.final_price,0)/sd.length).toFixed(2):"—";
                  const icons={walk_30:"",walk_60:"",drop_in:"",overnight:""};
                  return(
                    <div key={svc} style={{background:B.offWhite,border:`2.5px solid ${B.black}`,
                      borderRadius:16,padding:"14px 16px",boxShadow:`3px 3px 0 ${B.black}`}}>
                      <p style={{fontSize:20,marginBottom:4}}>{icons[svc]}</p>
                      <SectionLabel>{svc.replace("_"," ")}</SectionLabel>
                      <p style={{fontFamily:HEADING,fontSize:22,color:B.black}}>${avg}</p>
                      <p style={{fontFamily:BODY,fontSize:10,color:B.gray,marginBottom:8}}>avg price</p>
                      <div style={{display:"flex",gap:3,height:6,borderRadius:100,
                        overflow:"hidden",border:`1.5px solid ${B.black}22`}}>
                        {[[high,B.red],[med,B.amber],[std,B.ice]].map(([n,c],i)=>(
                          <div key={i} style={{flex:Math.max(n/tot,0.05),background:c}}/>
                        ))}
                      </div>
                      <p style={{fontFamily:HEADING,fontSize:9,color:B.gray,marginTop:6}}>
                        {Math.round(high/tot*100)}% HIGH · {Math.round(med/tot*100)}% MED</p>
                    </div>
                  );
                })}
              </div>
            </Card>
          </div>
        )}

        {/* WALKERS */}
        {tab==="walkers"&&(
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>
            <Card>
              <SectionLabel> Walker Leaderboard</SectionLabel>
              <WalkerTable walkers={walkers} loading={loading}/>
            </Card>
            <Card>
              <SectionLabel> Walk Volume</SectionLabel>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={[...walkers].sort((a,b)=>b.total_walks-a.total_walks)
                  .map(w=>({name:w.name.split(" ")[0],walks:w.total_walks,rating:w.rating}))}
                  layout="vertical" margin={{top:4,right:50,bottom:0,left:0}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={B.grayL} horizontal={false}/>
                  <XAxis type="number" tick={{fontFamily:HEADING,fontSize:9,fill:B.gray}}
                    tickLine={false} axisLine={false}/>
                  <YAxis dataKey="name" type="category"
                    tick={{fontFamily:HEADING,fontSize:10,fill:B.black}}
                    tickLine={false} axisLine={false} width={48}/>
                  <Tooltip content={({active,payload})=>active&&payload?.length?(
                    <div style={{background:B.white,border:`2px solid ${B.black}`,
                      borderRadius:10,padding:"8px 12px",boxShadow:`2px 2px 0 ${B.black}`}}>
                      <p style={{fontFamily:HEADING,fontSize:12}}>
                        {payload[0].value.toLocaleString()} walks</p>
                    </div>):null}/>
                  <Bar dataKey="walks" radius={[0,6,6,0]} maxBarSize={20}>
                    {[...walkers].sort((a,b)=>b.total_walks-a.total_walks)
                      .map((w,i)=><Cell key={i} fill={w.rating>=4.8?B.amber:B.ice}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        )}

        {/* PREDICT */}
        {tab==="predict"&&(
          <div style={{maxWidth:600,display:"flex",flexDirection:"column",gap:20}}>
            <Card>
              <SectionLabel> Live ML Predictor</SectionLabel>
              <LivePredictor apiOnline={apiOnline}/>
            </Card>
            <Card>
              <SectionLabel> API Endpoints</SectionLabel>
              {[["POST","/predict/demand","Demand forecast",B.iceDark,B.iceL],
                ["POST","/predict/price","Surge + price",B.red,B.redL],
                ["POST","/predict/demand/heatmap","Week heatmap",B.iceDark,B.iceL],
                ["GET", "/predict/price/schedule","Price schedule",B.amber,B.amberL],
                ["GET", "/bookings/summary","KPI summary",B.amber,B.amberL],
                ["GET", "/walkers","Walkers",B.iceDark,B.iceL],
              ].map(([method,path,desc,c,bg],i)=>(
                <div key={path}>
                  <div style={{display:"flex",alignItems:"center",gap:10,fontSize:12,padding:"9px 0"}}>
                    <div style={{background:bg,border:`2px solid ${c}`,borderRadius:100,
                      padding:"2px 8px",fontFamily:HEADING,fontSize:9,color:c,flexShrink:0}}>
                      {method}</div>
                    <span style={{fontFamily:"'Courier New',monospace",fontSize:11,
                      color:B.black,flex:1}}>{path}</span>
                    <span style={{color:B.gray,fontSize:11}}>{desc}</span>
                  </div>
                  {i<5&&<PawDivider/>}
                </div>
              ))}
            </Card>
            <div style={{textAlign:"center"}}>
              <span style={{fontFamily:HEADING,fontSize:10,color:B.gray,letterSpacing:"0.15em"}}>
                RELIABLE · FRIENDLY · LOCAL · ML-POWERED 
              </span>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
