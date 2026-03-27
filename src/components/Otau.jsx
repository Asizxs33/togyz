import React from 'react';

export const Otau = ({ index, count, isEnabled, isTuzdyk, owner, onClick, label, isHighlighted, animPhase }) => {
    let className = "otau wood-carved";
    if (isEnabled && count > 0) className += " enabled";
    else className += " disabled";
    
    if (isTuzdyk) {
        className += owner === 0 ? " tuzdyk-p1" : " tuzdyk-p2";
    }

    // Animation highlight classes
    if (isHighlighted) {
        if (animPhase === 'pickup') className += " anim-pickup";
        else if (animPhase === 'sow') className += " anim-sow";
        else if (animPhase === 'capture') className += " anim-capture";
        else if (animPhase === 'tuzdyk') className += " anim-tuzdyk";
    }

    // Generate stone elements - 2 columns neatly
    const stones = [];
    const maxVisibleStones = Math.min(count, 14);
    for (let i = 0; i < maxVisibleStones; i++) {
        stones.push(<div key={i} className="kumalak" />);
    }

    return (
        <div className="otau-wrapper">
             <div className="pocket-label top-label">{index >= 9 ? label : ''}</div>
             
             <div className={className} onClick={() => (isEnabled && count > 0) ? onClick(index) : null}>
                 {isTuzdyk ? (
                     <div className="tuzdyk-flag">
                          <span className="tuzdyk-icon">⭐</span>
                     </div>
                 ) : (
                     <div className="kumalak-grid">
                         {stones}
                         {count > 14 && <div className="plus-more">+{count - 14}</div>}
                     </div>
                 )}
                 <div className="count-badge">{isTuzdyk ? 'Т' : count}</div>
             </div>

             <div className="pocket-label bottom-label">{index < 9 ? label : ''}</div>
        </div>
    );
};
