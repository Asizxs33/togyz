import React from 'react';

export const Kazan = ({ player, count, isActive }) => {
    const className = `kazan wood-carved kazan-p${player + 1} ${isActive ? `active-p${player + 1}` : ''}`;
    
    // Generate stones line for horizontal kazan
    const stones = [];
    const maxVisibleStones = Math.min(count, 40); // Horizontal kazans can hold a long line
    for (let i = 0; i < maxVisibleStones; i++) {
        stones.push(<div key={i} className="kumalak kazan-kumalak" />);
    }

    return (
        <div className="kazan-wrapper">
             {/* Score Box on the left (or right depending on layout) */}
             <div className="kazan-score-box wood-carved">
                  <span className="kazan-score-number">{count}</span>
                  {/* Small decorative corner accents could go here in CSS */}
             </div>
             
             {/* The long pit itself */}
             <div className={className}>
                 <div className="kumalak-line">
                     {stones}
                     {count > 40 && <div className="plus-more kazan-plus">+{count - 40}</div>}
                 </div>
             </div>
        </div>
    );
};
