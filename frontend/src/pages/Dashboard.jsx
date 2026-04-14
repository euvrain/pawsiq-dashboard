import { useState, useEffect, useCallback } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, Cell, CartesianGrid,
} from "recharts";

// ── Design tokens ─────────────────────────────────────────────────────────────
const C = {
  cream:  "#F7F5F0",
  ink:    "#141810",
  sage:   "#3D6B4F",
  sageL:  "#EBF2EC",
  bark:   "#C4A882",
  barkL:  "#F5EFE5",
  warn:   "#D4622A",
  warnL:  "#FDF0EA",
  stone:  "#6B7063",
  rule:   "#E4E1D8",
  white:  "#FFFFFF",
};

const API_BASE = "http://localhost:8000";

const api = {
  demandHeatmap:  (month, year) =>
    fetch(`${API_BASE}/predict/demand/heatmap?month=${month}&year=${year}`).then(r => r.json()),
  pricingSchedule: (month) =>
    fetch(`${API_BASE}/predict/price/schedule?month=${month}`).then(r => r.json()),
  bookingSummary:  () =>
    fetch(`${API_BASE}/bookings/summary`).then(r => r.json()),
  walkers:         () =>
    fetch(`${API_BASE}/walkers`).then(r => r.json()),
  predictDemand:   (body) =>
    fetch(`${API_BASE}/predict/demand`, { method:"POST",
      headers:{"Content-Type":"application/json"}, body:JSON.stringify(body) }).then(r=>r.json()),
  predictPrice:    (body) =>
    fetch(`${API_BASE}/predict/price`, { method:"POST",
      headers:{"Content-Type":"application/json"}, body:JSON.stringify(body) }).then(r=>r.json()),
};

// ── Mock data (shown when API is offline) ─────────────────────────────────────
const MOCK = {
  summary: {
    total_bookings:9935, completed_bookings:9279, completion_rate_pct:93.4,
    total_revenue:240963.41, avg_surge:1.1873, peak_hour_pct:58.4,
    peak_avg_price:27.94, offpeak_avg_price:23.21, revenue_lift_vs_flat_pct:18.7,
    monthly_revenue:{"2024-07":9823,"2024-08":10241,"2024-09":11043,
                     "2024-10":10789,"2024-11":9654,"2024-12":9112},
  },
  walkers:[
    {user_id:"W0001",name:"Amara Stevens", rating:4.4,total_walks:908},
    {user_id:"W0002",name:"Devon Park",    rating:4.6,total_walks:1287},
    {user_id:"W0003",name:"Cleo Mitchell", rating:4.6,total_walks:1311},
    {user_id:"W0004",name:"Jordan Rivera", rating:4.8,total_walks:882},
    {user_id:"W0005",name:"Priya Okafor",  rating:4.8,total_walks:1339},
    {user_id:"W0006",name:"Chris Laurent", rating:4.4,total_walks:1315},
    {user_id:"W0007",name:"Natalie Diaz",  rating:4.8,total_walks:1274},
    {user_id:"W0008",name:"Marcus Chen",   rating:4.9,total_walks:963},
  ],
  heatmap: (() => {
    const rows=[], hw={7:1.8,8:1.9,9:1.7,10:0.9,11:0.9,12:0.8,13:0.8,
                        14:0.7,15:0.7,16:0.9,17:1.5,18:1.6,19:1.4,20:0.6};
    const dw=[1.2,1.2,1.2,1.2,1.2,0.8,0.8];
    for(let d=0;d<7;d++) for(let h=6;h<=20;h++)
      rows.push({day_of_week:d,hour_of_day:h,
        predicted:+(dw[d]*(hw[h]||0.6)*0.9).toFixed(2),
        is_peak:[7,8,9,17,18,19].includes(h)});
    return rows;
  })(),
  pricing: (() => {
    const rows=[];
    for(let d=0;d<7;d++) for(let h=6;h<=20;h++) {
      const isPeak=[7,8,9,17,18,19].includes(h), isWknd=d>=5;
      const s=Math.min(1.35,0.9+(isPeak?0.3:0.1)+(!isWknd?0.05:0)+Math.random()*0.05);
      rows.push({day_of_week:d,hour_of_day:h,service_type:"walk_30",
        surge:+s.toFixed(4),final_price:+(16*s).toFixed(2)});
    }
    return rows;
  })(),
};

// ── Shared components ─────────────────────────────────────────────────────────
const mono = {fontFamily:"'DM Mono',monospace"};
const Label = ({children,color=C.stone}) => (
  <p style={{...mono,fontSize:10,letterSpacing:"0.18em",textTransform:"uppercase",
    color,marginBottom:8}}>{children}</p>
);
const Card = ({children,style={}}) => (
  <div style={{background:C.white,border:`1px solid ${C.rule}`,
    borderRadius:14,padding:"20px 22px",...style}}>{children}</div>
);
const KpiCard = ({label,value,sub,accent=false}) => (
  <Card style={{flex:1,minWidth:140}}>
    <Label>{label}</Label>
    <p style={{fontFamily:"'Playfair Display',serif",fontSize:26,fontWeight:700,
      color:accent?C.sage:C.ink,lineHeight:1}}>{value}</p>
    {sub && <p style={{fontSize:11,color:C.stone,marginTop:4}}>{sub}</p>}
  </Card>
);
const Spinner = () => (
  <div style={{display:"flex",alignItems:"center",justifyContent:"center",
    height:120,color:C.stone,...mono,fontSize:11,letterSpacing:"0.1em"}}>LOADING...</div>
);
const Badge = ({label,color=C.sage,bg=C.sageL}) => (
  <span style={{...mono,fontSize:9,letterSpacing:"0.14em",textTransform:"uppercase",
    color,background:bg,padding:"3px 8px",borderRadius:4}}>{label}</span>
);

// ── Demand Heatmap ────────────────────────────────────────────────────────────
const DOW=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];
const HOURS=Array.from({length:15},(_,i)=>i+6);

function DemandHeatmap({data,loading}) {
  const [sel,setSel]=useState(null);
  if(loading) return <Spinner/>;
  const grid={};
  data.forEach(d=>{if(!grid[d.day_of_week])grid[d.day_of_week]={};
    grid[d.day_of_week][d.hour_of_day]=d.predicted;});
  const max=Math.max(...data.map(d=>d.predicted));
  const cellColor=v=>{const t=v/max;
    return t>0.75?C.warn:t>0.5?C.bark:t>0.25?"#9DB89F":C.sageL;};
  return (
    <div>
      <div style={{overflowX:"auto"}}>
        <table style={{borderCollapse:"collapse",width:"100%",minWidth:560}}>
          <thead><tr>
            <th style={{width:36,...mono,fontSize:9,color:C.stone,fontWeight:400,paddingBottom:6}}/>
            {HOURS.map(h=>(
              <th key={h} style={{...mono,fontSize:9,color:C.stone,fontWeight:400,
                paddingBottom:6,textAlign:"center",width:38}}>{h}:00</th>
            ))}
          </tr></thead>
          <tbody>{Array.from({length:7},(_,d)=>(
            <tr key={d}>
              <td style={{...mono,fontSize:9,color:C.stone,paddingRight:8,paddingBottom:3}}>
                {DOW[d]}</td>
              {HOURS.map(h=>{
                const val=grid[d]?.[h]??0,isSel=sel?.d===d&&sel?.h===h;
                return(
                  <td key={h} onMouseEnter={()=>setSel({d,h,val})}
                    onMouseLeave={()=>setSel(null)}
                    style={{padding:"2px 2px",cursor:"pointer"}}>
                    <div style={{width:34,height:22,borderRadius:4,background:cellColor(val),
                      border:isSel?`2px solid ${C.ink}`:"2px solid transparent",
                      transition:"all 0.1s",display:"flex",alignItems:"center",justifyContent:"center"}}>
                      {isSel&&<span style={{...mono,fontSize:8,color:C.ink,fontWeight:500}}>
                        {val.toFixed(1)}</span>}
                    </div>
                  </td>);
              })}
            </tr>
          ))}</tbody>
        </table>
      </div>
      <div style={{display:"flex",gap:12,marginTop:14,alignItems:"center"}}>
        <span style={{...mono,fontSize:9,color:C.stone}}>LOW</span>
        {[C.sageL,"#9DB89F",C.bark,C.warn].map((c,i)=>(
          <div key={i} style={{width:20,height:12,borderRadius:3,background:c}}/>
        ))}
        <span style={{...mono,fontSize:9,color:C.stone}}>HIGH</span>
        {sel&&<span style={{...mono,fontSize:10,color:C.ink,marginLeft:8}}>
          {DOW[sel.d]} {sel.h}:00 → {sel.val.toFixed(2)} predicted bookings</span>}
      </div>
    </div>
  );
}

// ── Pricing Panel ─────────────────────────────────────────────────────────────
function PricingPanel({data,loading}) {
  if(loading) return <Spinner/>;
  const monWalk=data.filter(d=>d.day_of_week===0&&d.service_type==="walk_30")
    .sort((a,b)=>a.hour_of_day-b.hour_of_day)
    .map(d=>({hour:`${d.hour_of_day}:00`,surge:d.surge,price:d.final_price,
      fill:d.surge>=1.25?C.warn:d.surge>=1.10?C.bark:C.sage}));
  return(
    <div>
      <p style={{fontSize:12,color:C.stone,marginBottom:14}}>
        30-min walk · Monday · Ridge Regression model</p>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={monWalk} margin={{top:4,right:4,bottom:0,left:-10}}>
          <CartesianGrid strokeDasharray="3 3" stroke={C.rule} vertical={false}/>
          <XAxis dataKey="hour" tick={{...mono,fontSize:9,fill:C.stone}} tickLine={false} axisLine={false}/>
          <YAxis tick={{...mono,fontSize:9,fill:C.stone}} tickLine={false} axisLine={false}
            domain={[0.8,1.4]} tickFormatter={v=>`×${v.toFixed(2)}`}/>
          <Tooltip content={({active,payload})=>active&&payload?.length?(
            <div style={{background:C.white,border:`1px solid ${C.rule}`,borderRadius:8,
              padding:"10px 14px",...mono,fontSize:11}}>
              <p style={{color:C.ink}}>{payload[0].payload.hour}</p>
              <p style={{color:C.sage}}>×{payload[0].value.toFixed(4)}</p>
              <p style={{color:C.stone}}>${payload[0].payload.price}</p>
            </div>):null}/>
          <Bar dataKey="surge" radius={[4,4,0,0]} maxBarSize={28}>
            {monWalk.map((d,i)=><Cell key={i} fill={d.fill}/>)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div style={{display:"flex",gap:16,marginTop:10}}>
        {[["high ×1.25+",C.warn,C.warnL],["medium ×1.10+",C.bark,C.barkL],
          ["standard",C.sage,C.sageL]].map(([l,c,bg])=>(
          <Badge key={l} label={l} color={c} bg={bg}/>
        ))}
      </div>
    </div>
  );
}

// ── Walker Table ──────────────────────────────────────────────────────────────
function WalkerTable({walkers,loading}) {
  if(loading) return <Spinner/>;
  const sorted=[...walkers].sort((a,b)=>b.total_walks-a.total_walks);
  const maxW=Math.max(...sorted.map(w=>w.total_walks));
  return(
    <div style={{display:"flex",flexDirection:"column",gap:10}}>
      {sorted.map(w=>(
        <div key={w.user_id} style={{display:"flex",alignItems:"center",gap:12}}>
          <div style={{width:32,height:32,borderRadius:"50%",background:C.sage,
            color:C.white,flexShrink:0,display:"flex",alignItems:"center",
            justifyContent:"center",...mono,fontSize:11}}>{w.name[0]}</div>
          <div style={{flex:1,minWidth:0}}>
            <div style={{display:"flex",justifyContent:"space-between",
              alignItems:"center",marginBottom:3}}>
              <span style={{fontSize:13,fontWeight:500}}>{w.name}</span>
              <span style={{...mono,fontSize:10,color:C.stone}}>
                ★{w.rating} · {w.total_walks.toLocaleString()}</span>
            </div>
            <div style={{height:4,borderRadius:2,background:C.rule,overflow:"hidden"}}>
              <div style={{height:"100%",borderRadius:2,
                background:w.rating>=4.8?C.sage:C.bark,
                width:`${(w.total_walks/maxW)*100}%`,transition:"width 0.6s ease"}}/>
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
  const DOW_L=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];

  const predict=async()=>{
    setLoading(true);
    try {
      if(apiOnline){
        const [demand,price]=await Promise.all([
          api.predictDemand({hour_of_day:hour,day_of_week:dow,month:4,year:2025}),
          api.predictPrice({hour_of_day:hour,day_of_week:dow,month:4,
            service_type:svc,zip_hour_demand:2}),
        ]);
        setResult({demand,price});
      } else {
        const isPeak=[7,8,9,17,18,19].includes(hour),isWknd=dow>=5;
        const surge=Math.min(1.35,0.9+(isPeak?0.3:0.1)+(!isWknd?0.05:0));
        setResult({
          demand:{predicted_bookings:isPeak?2.1:0.6,
            demand_level:isPeak?"medium":"low",is_peak_hour:isPeak},
          price:{surge_multiplier:+surge.toFixed(4),base_price:BASE[svc],
            final_price:+(BASE[svc]*surge).toFixed(2),
            pricing_tier:surge>=1.25?"high":surge>=1.10?"medium":"standard",
            price_explanation:isPeak?"Peak hour — surge applied":"Standard pricing"},
        });
      }
    } catch(e){console.error(e);}
    setLoading(false);
  };

  const selStyle={background:C.cream,border:`1px solid ${C.rule}`,borderRadius:8,
    padding:"7px 10px",fontSize:12,color:C.ink,cursor:"pointer",
    fontFamily:"'DM Sans',sans-serif",outline:"none"};

  return(
    <div>
      <p style={{fontSize:12,color:C.stone,marginBottom:14}}>
        Calls both ML endpoints simultaneously. Try different time slots.</p>
      <div style={{display:"flex",gap:10,flexWrap:"wrap",marginBottom:16}}>
        <select value={hour} onChange={e=>{setHour(+e.target.value);setResult(null);}}
          style={selStyle}>
          {HOURS.map(h=><option key={h} value={h}>{h}:00</option>)}
        </select>
        <select value={dow} onChange={e=>{setDow(+e.target.value);setResult(null);}}
          style={selStyle}>
          {DOW_L.map((d,i)=><option key={i} value={i}>{d}</option>)}
        </select>
        <select value={svc} onChange={e=>{setSvc(e.target.value);setResult(null);}}
          style={selStyle}>
          <option value="walk_30">30-min Walk</option>
          <option value="walk_60">60-min Walk</option>
          <option value="drop_in">Drop-in</option>
          <option value="overnight">Overnight</option>
        </select>
        <button onClick={predict} disabled={loading} style={{
          background:loading?C.rule:C.ink,color:C.white,border:"none",
          borderRadius:8,padding:"7px 18px",fontSize:12,
          cursor:loading?"not-allowed":"pointer",
          fontFamily:"'DM Sans',sans-serif",transition:"background 0.15s"}}>
          {loading?"...":"Predict →"}
        </button>
      </div>
      {result&&(
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
          <div style={{background:C.sageL,borderRadius:10,padding:"14px 16px",
            border:`1px solid ${C.sage}33`}}>
            <Label color={C.sage}>Demand Forecast</Label>
            <p style={{fontFamily:"'Playfair Display',serif",fontSize:24,
              fontWeight:700,color:C.sage}}>
              {result.demand.predicted_bookings.toFixed(2)}</p>
            <p style={{fontSize:11,color:C.stone,marginTop:2}}>
              bookings/slot · <strong>{result.demand.demand_level}</strong>
              {result.demand.is_peak_hour&&" · ⚡ peak"}</p>
          </div>
          <div style={{background:result.price.pricing_tier==="high"?C.warnL:C.sageL,
            borderRadius:10,padding:"14px 16px",
            border:`1px solid ${result.price.pricing_tier==="high"?C.warn+"33":C.sage+"33"}`}}>
            <Label color={result.price.pricing_tier==="high"?C.warn:C.sage}>
              Dynamic Price</Label>
            <p style={{fontFamily:"'Playfair Display',serif",fontSize:24,fontWeight:700,
              color:result.price.pricing_tier==="high"?C.warn:C.sage}}>
              ${result.price.final_price}</p>
            <p style={{fontSize:11,color:C.stone,marginTop:2}}>
              ×{result.price.surge_multiplier} · {result.price.price_explanation}</p>
          </div>
        </div>
      )}
      {!apiOnline&&<p style={{...mono,fontSize:9,color:C.stone,marginTop:10,
        letterSpacing:"0.1em"}}>⚠ API OFFLINE — MOCK PREDICTIONS</p>}
    </div>
  );
}

// ── Revenue Chart ─────────────────────────────────────────────────────────────
function RevenueChart({data}) {
  const chartData=Object.entries(data).map(([k,v])=>({month:k.slice(5),revenue:v}));
  return(
    <ResponsiveContainer width="100%" height={120}>
      <BarChart data={chartData} margin={{top:4,right:4,bottom:0,left:-10}}>
        <CartesianGrid strokeDasharray="3 3" stroke={C.rule} vertical={false}/>
        <XAxis dataKey="month" tick={{...mono,fontSize:9,fill:C.stone}}
          tickLine={false} axisLine={false}/>
        <YAxis tick={{...mono,fontSize:9,fill:C.stone}} tickLine={false} axisLine={false}
          tickFormatter={v=>`$${(v/1000).toFixed(0)}k`}/>
        <Tooltip content={({active,payload})=>active&&payload?.length?(
          <div style={{background:C.white,border:`1px solid ${C.rule}`,
            borderRadius:8,padding:"8px 12px",...mono,fontSize:11}}>
            <p style={{color:C.sage}}>${payload[0].value.toLocaleString()}</p>
          </div>):null}/>
        <Bar dataKey="revenue" fill={C.sage} radius={[4,4,0,0]} maxBarSize={32}/>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────
export default function Dashboard() {
  const [tab,setTab]             = useState("overview");
  const [summary,setSummary]     = useState(null);
  const [walkers,setWalkers]     = useState([]);
  const [heatmap,setHeatmap]     = useState([]);
  const [pricing,setPricing]     = useState([]);
  const [apiOnline,setApiOnline] = useState(false);
  const [loading,setLoading]     = useState(true);
  const [fadeIn,setFadeIn]       = useState(false);

  useEffect(()=>{
    const load=async()=>{
      setLoading(true);
      try {
        const [sum,walk,heat,price]=await Promise.all([
          api.bookingSummary(), api.walkers(),
          api.demandHeatmap(4,2025), api.pricingSchedule(4),
        ]);
        setSummary(sum); setWalkers(walk);
        setHeatmap(heat.heatmap); setPricing(price.schedule);
        setApiOnline(true);
      } catch {
        setSummary(MOCK.summary); setWalkers(MOCK.walkers);
        setHeatmap(MOCK.heatmap); setPricing(MOCK.pricing);
        setApiOnline(false);
      }
      setLoading(false);
      setTimeout(()=>setFadeIn(true),50);
    };
    load();
  },[]);

  const TABS=["overview","demand","pricing","walkers","predict"];

  return(
    <div style={{minHeight:"100vh",background:C.cream,fontFamily:"'DM Sans',sans-serif"}}>
      <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet"/>

      {/* Header */}
      <header style={{background:C.ink,padding:"20px 40px",display:"flex",
        alignItems:"center",justifyContent:"space-between"}}>
        <div>
          <p style={{...mono,fontSize:10,letterSpacing:"0.2em",color:C.bark,marginBottom:4}}>
            PAWSIQ · OPERATOR</p>
          <h1 style={{fontFamily:"'Playfair Display',serif",fontSize:24,
            color:C.white,fontWeight:700,margin:0}}>ML Intelligence Dashboard</h1>
        </div>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          <div style={{width:7,height:7,borderRadius:"50%",
            background:apiOnline?"#6FCF97":C.warn}}/>
          <span style={{...mono,fontSize:10,color:apiOnline?"#6FCF97":C.warn,
            letterSpacing:"0.1em"}}>{apiOnline?"API LIVE":"MOCK DATA"}</span>
        </div>
      </header>

      {/* Tabs */}
      <nav style={{background:C.white,borderBottom:`1px solid ${C.rule}`,
        padding:"0 40px",display:"flex",gap:0}}>
        {TABS.map(t=>(
          <button key={t} onClick={()=>setTab(t)} style={{
            ...mono,fontSize:10,letterSpacing:"0.14em",textTransform:"uppercase",
            padding:"14px 20px",background:"none",border:"none",cursor:"pointer",
            color:tab===t?C.ink:C.stone,
            borderBottom:tab===t?`2px solid ${C.ink}`:"2px solid transparent",
            transition:"all 0.15s"}}>{t}</button>
        ))}
      </nav>

      {/* Content */}
      <main style={{padding:"32px 40px",maxWidth:1200,margin:"0 auto",
        opacity:fadeIn?1:0,transition:"opacity 0.4s"}}>

        {/* Overview */}
        {tab==="overview"&&(
          <div style={{display:"flex",flexDirection:"column",gap:24}}>
            <div style={{display:"flex",gap:14,flexWrap:"wrap"}}>
              <KpiCard label="Total Revenue"
                value={summary?`$${(summary.total_revenue/1000).toFixed(0)}k`:"—"}
                sub="2-year synthetic dataset" accent/>
              <KpiCard label="Completed Bookings"
                value={summary?.completed_bookings?.toLocaleString()??"—"}
                sub={`${summary?.completion_rate_pct??"—"}% completion rate`}/>
              <KpiCard label="Avg Surge"
                value={summary?`×${summary.avg_surge.toFixed(3)}`:"—"}
                sub={`+${summary?.revenue_lift_vs_flat_pct??"—"}% vs flat rate`} accent/>
              <KpiCard label="Peak Hour %"
                value={summary?`${summary.peak_hour_pct}%`:"—"}
                sub={`$${summary?.peak_avg_price??"—"} vs $${summary?.offpeak_avg_price??"—"} off-peak`}/>
            </div>
            <Card>
              <Label>Monthly Revenue — Last 6 Months</Label>
              {summary?<RevenueChart data={summary.monthly_revenue}/>:<Spinner/>}
            </Card>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
              <Card>
                <Label>Demand Model · GradientBoosting</Label>
                <div style={{display:"flex",flexDirection:"column",gap:8,marginTop:4}}>
                  {[["MAE","0.738 bookings/slot"],["RMSE","0.912"],
                    ["R²","0.098"],["RMSE vs baseline","+5.0%"]].map(([k,v])=>(
                    <div key={k} style={{display:"flex",justifyContent:"space-between",
                      fontSize:13,borderBottom:`1px solid ${C.rule}`,paddingBottom:6}}>
                      <span style={{color:C.stone}}>{k}</span>
                      <span style={{fontWeight:500}}>{v}</span>
                    </div>
                  ))}
                </div>
              </Card>
              <Card>
                <Label>Pricing Model · Ridge Regression</Label>
                <div style={{display:"flex",flexDirection:"column",gap:8,marginTop:4}}>
                  {[["MAE","0.0569 ×surge"],["RMSE","0.0679"],
                    ["R²","0.727"],["Revenue lift","+17.1%"]].map(([k,v])=>(
                    <div key={k} style={{display:"flex",justifyContent:"space-between",
                      fontSize:13,borderBottom:`1px solid ${C.rule}`,paddingBottom:6}}>
                      <span style={{color:C.stone}}>{k}</span>
                      <span style={{fontWeight:500,
                        color:k==="Revenue lift"?C.sage:C.ink}}>{v}</span>
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          </div>
        )}

        {/* Demand */}
        {tab==="demand"&&(
          <div style={{display:"flex",flexDirection:"column",gap:24}}>
            <Card>
              <Label>Demand Heatmap — April 2025 · XGBoost model</Label>
              <p style={{fontSize:12,color:C.stone,marginBottom:16}}>
                Predicted bookings per hour slot. Hover a cell to see the value.
                Morning (7–9am) and evening (5–7pm) peaks learned from 2 years of NJ data.</p>
              <DemandHeatmap data={heatmap} loading={loading}/>
            </Card>
            <Card>
              <Label>Avg Predicted Demand by Hour</Label>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={HOURS.map(h=>({
                  hour:`${h}:00`,
                  avg:heatmap.filter(d=>d.hour_of_day===h)
                      .reduce((s,d)=>s+d.predicted,0)/
                      Math.max(1,heatmap.filter(d=>d.hour_of_day===h).length),
                }))} margin={{top:4,right:4,bottom:0,left:-10}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.rule} vertical={false}/>
                  <XAxis dataKey="hour" tick={{...mono,fontSize:9,fill:C.stone}}
                    tickLine={false} axisLine={false}/>
                  <YAxis tick={{...mono,fontSize:9,fill:C.stone}}
                    tickLine={false} axisLine={false}/>
                  <Tooltip content={({active,payload})=>active&&payload?.length?(
                    <div style={{background:C.white,border:`1px solid ${C.rule}`,
                      borderRadius:8,padding:"8px 12px",...mono,fontSize:11}}>
                      <p>{payload[0].value.toFixed(2)} avg bookings</p>
                    </div>):null}/>
                  <Bar dataKey="avg" radius={[4,4,0,0]} maxBarSize={30}>
                    {HOURS.map((h,i)=>
                      <Cell key={i} fill={[7,8,9,17,18,19].includes(h)?C.warn:C.sage}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        )}

        {/* Pricing */}
        {tab==="pricing"&&(
          <div style={{display:"flex",flexDirection:"column",gap:24}}>
            <Card>
              <Label>Surge by Hour · Ridge Regression · Monday walk_30</Label>
              <PricingPanel data={pricing} loading={loading}/>
            </Card>
            <Card>
              <Label>Pricing Tier Breakdown — All Services</Label>
              <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:12,marginTop:8}}>
                {["walk_30","walk_60","drop_in","overnight"].map(svc=>{
                  const sd=pricing.filter(d=>d.service_type===svc);
                  const high=sd.filter(d=>d.surge>=1.25).length;
                  const med=sd.filter(d=>d.surge>=1.10&&d.surge<1.25).length;
                  const std=sd.filter(d=>d.surge<1.10).length;
                  const tot=sd.length||1;
                  const BASE={walk_30:16,walk_60:24,drop_in:14,overnight:55};
                  const avgRev=sd.length?(sd.reduce((s,d)=>s+d.final_price,0)/sd.length).toFixed(2):"—";
                  return(
                    <div key={svc} style={{background:C.cream,borderRadius:10,
                      padding:"14px 16px",border:`1px solid ${C.rule}`}}>
                      <Label>{svc.replace("_"," ")}</Label>
                      <p style={{fontFamily:"'Playfair Display',serif",
                        fontSize:20,fontWeight:700}}>${avgRev}</p>
                      <p style={{fontSize:10,color:C.stone,marginBottom:8}}>avg price</p>
                      <div style={{display:"flex",gap:4}}>
                        {[["H",high,C.warn],["M",med,C.bark],["S",std,C.sage]].map(([l,n,c])=>(
                          <div key={l} style={{flex:Math.max(n/tot,0.05),height:4,
                            borderRadius:2,background:c}}/>
                        ))}
                      </div>
                      <p style={{...mono,fontSize:9,color:C.stone,marginTop:6}}>
                        {Math.round(high/tot*100)}% high · {Math.round(med/tot*100)}% med</p>
                    </div>
                  );
                })}
              </div>
            </Card>
          </div>
        )}

        {/* Walkers */}
        {tab==="walkers"&&(
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:24}}>
            <Card>
              <Label>Walker Volume & Rating</Label>
              <WalkerTable walkers={walkers} loading={loading}/>
            </Card>
            <Card>
              <Label>Volume Distribution</Label>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={[...walkers].sort((a,b)=>b.total_walks-a.total_walks)
                  .map(w=>({name:w.name.split(" ")[0],walks:w.total_walks}))}
                  layout="vertical" margin={{top:4,right:40,bottom:0,left:0}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.rule} horizontal={false}/>
                  <XAxis type="number" tick={{...mono,fontSize:9,fill:C.stone}}
                    tickLine={false} axisLine={false}/>
                  <YAxis dataKey="name" type="category" tick={{...mono,fontSize:10,fill:C.stone}}
                    tickLine={false} axisLine={false} width={52}/>
                  <Tooltip content={({active,payload})=>active&&payload?.length?(
                    <div style={{background:C.white,border:`1px solid ${C.rule}`,
                      borderRadius:8,padding:"8px 12px",...mono,fontSize:11}}>
                      <p>{payload[0].value.toLocaleString()} walks</p>
                    </div>):null}/>
                  <Bar dataKey="walks" radius={[0,4,4,0]} maxBarSize={22}>
                    {[...walkers].sort((a,b)=>b.total_walks-a.total_walks)
                      .map((w,i)=><Cell key={i} fill={w.rating>=4.8?C.sage:C.bark}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        )}

        {/* Live Predict */}
        {tab==="predict"&&(
          <div style={{maxWidth:600}}>
            <Card>
              <Label>Live ML Predictor</Label>
              <LivePredictor apiOnline={apiOnline}/>
            </Card>
            <Card style={{marginTop:24}}>
              <Label>API Endpoints</Label>
              <div style={{display:"flex",flexDirection:"column",gap:0,marginTop:4}}>
                {[["POST","/predict/demand","Booking demand forecast"],
                  ["POST","/predict/price","Surge multiplier + final price"],
                  ["POST","/predict/demand/heatmap","Full week × hour demand grid"],
                  ["GET", "/predict/price/schedule","Full pricing schedule"],
                  ["GET", "/bookings/summary","KPI summary"],
                  ["GET", "/walkers","Walker profiles"],
                ].map(([method,path,desc])=>(
                  <div key={path} style={{display:"flex",alignItems:"center",
                    gap:10,fontSize:12,padding:"9px 0",borderBottom:`1px solid ${C.rule}`}}>
                    <Badge label={method}
                      color={method==="POST"?C.warn:C.sage}
                      bg={method==="POST"?C.warnL:C.sageL}/>
                    <span style={{...mono,fontSize:11,color:C.ink,flex:1}}>{path}</span>
                    <span style={{color:C.stone,fontSize:11}}>{desc}</span>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        )}
      </main>
    </div>
  );
}
